// 半精度类型（__half）、数学函数
#include <cuda_fp16.h>      // __half、__h* 系列函数
#include <math_constants.h> // FLT_MIN
#include <float.h>          // FLT_MIN（某些平台需要）
#include <cub/cub.cuh>      
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#if CCCL_VERSION >= 2008002
#include <cuda/std/functional>
#define CUB_REDUCTIONOP_MAX                                                                                            \
    cuda::maximum<> {}
#else
#define CUB_REDUCTIONOP_MAX cub::Max()
#endif

#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

// The maximum number of resident threads per SM varies by arch.
// For A100/H100 and all prior to Turing, it is 2048, which allows
// for 2 full blocks of 1024 threads per SM.
// Reference:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability
#if __CUDA_ARCH__ == 750
#define BNB_MAX_THREADS_PER_SM 1024
#elif __CUDA_ARCH__ >= 860 && __CUDA_ARCH__ <= 890
#define BNB_MAX_THREADS_PER_SM 1536
#else
#define BNB_MAX_THREADS_PER_SM 2048
#endif

// Maximum resident warps per SM is always directly related to the number of threads.
#define BNB_MAX_WARPS_PER_SM ((BNB_MAX_THREADS_PER_SM) / (BNB_WARP_SIZE))

// Maximum resident blocks per SM may vary.
#if __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870
#define BNB_MAX_BLOCKS_PER_SM 16
#else
#define BNB_MAX_BLOCKS_PER_SM ((BNB_MAX_WARPS_PER_SM) / 2)
#endif

//ref:cd /CUDA-Triton-Learn/third_party/bitsandbytes/csrc/kernels.cu

// 模板参数：
// T               —— 输入矩阵元素类型（通常是半/单精度，如 __half 或 float）。
// THREADS         —— 每个 block 启用的线程数（决定并行度和分条访问步幅）。
// SPARSE_DECOMP   —— 稀疏分解开关（编译期常量）：1 表示对“离群值(＞threshold)”不量化而置 0。
template <typename T, int THREADS, int SPARSE_DECOMP>

// __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
//   - 向编译器传达“本 kernel 以最多 1024 线程/块”为优化目标，并期望每个 SM 至少常驻 BNB_MAX_THREADS_PER_SM/1024 个 block。
//   - 编译器据此在“寄存器使用 vs. 并发(occupancy)”间做权衡，避免过多寄存器导致并发下降（或溢出到本地内存）。:contentReference[oaicite:0]{index=0}
__launch_bounds__(1024, BNB_MAX_THREADS_PER_SM / 1024) __global__
void kInt8VectorQuant(T* __restrict__ A,      // [rows, cols] 行主存放的输入矩阵
                      int8_t* out,            // [rows, cols] 行主存放的量化输出
                      float* rowStats,        // [rows] 逐行的统计量（这里存每行的 absmax）
                      float threshold,        // 稀疏分解中的离群值阈值
                      int rows, int cols) {

    // 对 Maxwell 架构(sm50/52) 且 CUDA<12.2 的旧环境，归约用 fp32 更安全；
    // 其他新环境可直接用 T（如 fp16）做归约，减少类型转换的代价。
    // 这是出于数值与平台兼容性的工程权衡。
#if (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR >= 2) || BNB_FP16_AVAILABLE
    using TReduction = T;
#else
    using TReduction = float;
#endif

    // 使用 CUB 的 block 级归约原语：所有线程在一个 block 内协作，做“最大值”归约。
    // TReduction 是参与归约的标量类型，THREADS 是 block 的线程数。:contentReference[oaicite:1]{index=1}
    using BlockReduceT = cub::BlockReduce<TReduction, THREADS>;

    // 【并行划分思想】
    //  - 采用“一行一个 block”的映射：blockIdx.x 对应 row_id。
    //  - 线程对该行做“条带式(strided)访问”：tid 访问 i = tid, tid+THREADS, tid+2*THREADS, ...
    //  - 先各自求“线程局部的 |x| 最大值”，再用 BlockReduce 求整行 absmax。
    //
    // 共享内存用于：
    //  - BlockReduce 的临时存储（CUB 需要）
    //  - 存放整行 absmax 以便后续所有线程复用（避免重复全局内存访问）
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    __shared__ TReduction smem_row_absmax;

    const int row_id   = blockIdx.x;          // 当前处理的行
    const T*  row_data = A + (row_id * cols); // 指向这一行的起始地址（行主存）

    // 每个线程在自己的条带上找“局部绝对值最大”
    // 这里初始化为最小浮点（负无穷，或很小的负值）以便后面用 fmaxf 更新
    TReduction row_local_absmax = -FLT_MIN;

    // 条带式读取：i 从 tid 开始，每次步进 THREADS
    for (int i = threadIdx.x; i < cols; i += THREADS) {
        // 读取并取绝对值：
        //  - __ldcs() 是带缓存提示的加载 intrinsic（streaming cache hint），可影响缓存行为；
        //    这里配合 fabsf 计算绝对值（若 T 为 fp16，前面把 TReduction 设为适当类型以保持数值稳健）。:contentReference[oaicite:2]{index=2}
        const TReduction absval = fabsf(__ldcs(&(row_data[i])));

        // 若启用“稀疏分解”（SPARSE_DECOMP==1）：
        //   - 超过阈值的值视为离群值（outlier），不参与本行 absmax 的统计（只在量化阶段置 0）。
        //   - 这样可避免离群值把量化 scale 拉大，提升非离群值的量化分辨率。
        if constexpr (SPARSE_DECOMP) {
            row_local_absmax = fmaxf(row_local_absmax,
                                     (absval < TReduction(threshold)) ? absval : row_local_absmax);
        } else {
            // 常规：全量参与，按绝对值更新最大
            row_local_absmax = fmaxf(row_local_absmax, absval);
        }
    }

    // 用 CUB 在 block 内做一次“最大值”归约，得到整行的 absmax。
    // Reduce(..., cub::Max(), cols) 的“cols”参数用于告知参与范围/边界（实现可忽略超界）。
    const TReduction row_absmax = BlockReduceT(temp_storage).Reduce(row_local_absmax, cub::Max(), cols);

    if (threadIdx.x == 0) {
        // 仅由 0 号线程把结果写入：
        //   - 全局 rowStats[row_id]（便于核外或后续核使用）
        //   - 共享内存 smem_row_absmax（本行内统一使用的 scale 分母）
        rowStats[row_id]   = smem_row_absmax = row_absmax;
    }

    // 栅栏：确保全体线程在看到 smem_row_absmax 之前完成其写入。:contentReference[oaicite:3]{index=3}
    __syncthreads();

    // 【量化阶段（逐行）】
    // 线性标度：scale = 127 / absmax
    // - 127 是有符号 int8 的最大正值（-128~127），用它把 [-absmax, +absmax] 线性映射到 [-127, +127]。
    // - __fdividef 是单精度的快速除法 intrinsic。:contentReference[oaicite:4]{index=4}
    const float scale = __fdividef(127.0f, smem_row_absmax);

    // 仍然采用条带式并行写出量化结果
    for (int i = threadIdx.x; i < cols; i += THREADS) {
        float val = row_data[i];  // 读原值（此处以 float 计算更直观；若 T=__half，编译器会做相应转换）

        if constexpr (SPARSE_DECOMP) {
            // 稀疏分解：离群值( |val|>=threshold ) 不量化，直接写 0（保持“稀疏 + 分离 outlier”策略）
            // 非离群值：乘以 scale 后用“就近取整”为 int（__float2int_rn）再存为 int8。:contentReference[oaicite:5]{index=5}
            out[row_id * cols + i] = (fabsf(val) < threshold)
                                        ? __float2int_rn(val * scale)
                                        : 0;
        } else {
            // 常规：所有元素都按线性标度量化为 int8
            out[row_id * cols + i] = __float2int_rn(val * scale);  // 四舍六入五成偶（round-to-nearest-even）
        }
    }
}