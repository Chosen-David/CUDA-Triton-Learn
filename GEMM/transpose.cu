#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//v0
// 矩阵转置用时：224.45 微秒
// 内存带宽：122.97 GB/s
// 带宽利用率：16.01%

// 获取 GPU 的理论带宽
float get_theoretical_bandwidth() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 假设使用设备 0
    return prop.memoryClockRate * 2.0f * prop.memoryBusWidth / 8 / 1e6; // GB/s
}

#define CHECK_CUDA_CALL(call)                                          \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA 错误发生在 %s:%d: %s\n", __FILE__,     \
                    __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// CUDA Kernel for matrix transpose
__global__ void mat_transpose_kernel_v0(const float* idata, float* odata, int M, int N) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x; // Thread's column index
    int ty = threadIdx.y + blockDim.y * blockIdx.y; // Thread's row index

    if (tx < N && ty < M) {
        odata[tx * M + ty] = idata[ty * N + tx]; // Transpose operation
    }
}

// Host function to launch the kernel
void mat_transpose_v0(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 16;
    dim3 block(BLOCK_SZ, BLOCK_SZ);
    dim3 grid((N + BLOCK_SZ - 1) / BLOCK_SZ, (M + BLOCK_SZ - 1) / BLOCK_SZ);
    mat_transpose_kernel_v0<<<grid, block>>>(idata, odata, M, N);
    CHECK_CUDA_CALL(cudaDeviceSynchronize()); // Ensure kernel execution completes
}


template <int BLOCK_SZ>
__global__ void mat_transpose_kernel_v1(const float* idata, float* odata, int M, int N) {
    const int tx=threadIdx.x,ty=threadIdx.y;
    const int bx=blockIdx.x,by=blockIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ];

    int x=tx+bx*BLOCK_SZ;
    int y=ty+by*BLOCK_SZ;

    if(x<N&&y<M){
        sdata[ty][tx]=idata[y*N+x];
    }
    __syncthreads();

    x=tx+by*BLOCK_SZ;
    y=ty+bx*BLOCK_SZ;
    if(x<M&&y<N){
        odata[y*M+x]=sdata[tx][ty];
    }
}
inline int Ceil(int x, int y) {
    return (x + y - 1) / y;
}
void mat_transpose_v1(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 16;
    dim3 block(BLOCK_SZ, BLOCK_SZ);
    dim3 grid(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    mat_transpose_kernel_v1<BLOCK_SZ><<<grid, block>>>(idata, odata, M, N);
}


template <int BLOCK_SZ>
__global__ void mat_transpose_kernel_v2(const float* idata, float* odata, int M, int N) {
    const int tx=threadIdx.x,ty=threadIdx.y;
    const int bx=blockIdx.x,by=blockIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ+1];

    int x=tx+bx*BLOCK_SZ;
    int y=ty+by*BLOCK_SZ;

    if(x<N&&y<M){
        sdata[ty][tx]=idata[y*N+x];
    }
    __syncthreads();

    x=tx+by*BLOCK_SZ;
    y=ty+bx*BLOCK_SZ;
    if(x<M&&y<N){
        odata[y*M+x]=sdata[tx][ty];
    }


}


void mat_transpose_v2(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 16;
    dim3 block(BLOCK_SZ, BLOCK_SZ);
    dim3 grid(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    mat_transpose_kernel_v2<BLOCK_SZ><<<grid, block>>>(idata, odata, M, N);
}


template<int BLOCK_SZ,int NUM_PER_THREAD>
__global__ void mat_transpose_kernel_v3(const float* idata, float* odata, int M, int N) {

    const int tx=threadIdx.x,ty=threadIdx.y;
    const int bx=blockIdx.x,by=blockIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ+1];

    int x=tx+bx*BLOCK_SZ;
    int y=ty+by*BLOCK_SZ;

    constexpr int ROW_STRIDE=BLOCK_SZ/NUM_PER_THREAD;

    if(x<N){
        #pragma unroll
        for(int y_off=0;y_off<BLOCK_SZ;y_off+=ROW_STRIDE){
            if(y+y_off<M){
                sdata[ty+y_off][tx]=idata[(y+y_off)*N+x];
            }
        }
    }


    __syncthreads();

    x=tx+by*BLOCK_SZ;
    y=ty+bx*BLOCK_SZ;
    if(x<M){
         for(int y_off=0;y_off<BLOCK_SZ;y_off+=ROW_STRIDE){
            if(y+y_off<M){
                odata[(y+y_off)*M+x]=sdata[tx][ty+y_off];
            }
         }
    }
}

void mat_transpose_v3(const float* idata, float* odata, int M, int N){
    constexpr int BLOCK_SZ=32;
    constexpr int NUM_PER_THREAD=4;
    dim3 block(BLOCK_SZ, BLOCK_SZ/NUM_PER_THREAD);
    dim3 grid(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    mat_transpose_kernel_v3<BLOCK_SZ,NUM_PER_THREAD><<<grid,block>>>(idata,odata,M,N);
}

#define FETCH_CFLOAT4(p) (reinterpret_cast<const float4>(&p)[0]);
#define FETCH_FLOAT4(p) (reinterpret_cast<float4>(&p)[0]);
template <int BLOCK_SZ>
__global__ void mat_transpose_kernel_v3_5(const float* idata, float* odata, int M, int N) {
    
    const int tx=threadIdx.x,ty=threadIdx.y;
    const int bx=blockIdx.x,by=blockIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ];

    int x=tx*4+bx*BLOCK_SZ;
    int y=ty+by*BLOCK_SZ;


   if(x<N&&y<M){
        FETCH_FLOAT4(sdata[ty][tx*4])=FETCH_CFLOAT4(idata[y*N+x]);
   }

    __syncthreads();

    x=tx*4+by*BLOCK_SZ;
    y=ty+bx*BLOCK_SZ;
    float tmp[4];
    if(x<M&&y<N){
        #pragma unroll
        for(int i=0;i<4;i++){
            tmp[i]=sdata[tx*4+i][i];
        }
        FETCH_FLOAT4(odata)=FETCH_FLOAT4(tmp);
    }
}
void mat_transpose_v3_5(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 32;
    dim3 block(BLOCK_SZ / 4, BLOCK_SZ);
    dim3 grid(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    mat_transpose_kernel_v3_5<BLOCK_SZ><<<grid, block>>>(idata, odata, M, N);
}




int main() {
    // 矩阵维度
    const int M = 2300;
    const int N = 1500;

    // 分配主机内存
    size_t size = M * N * sizeof(float);
    float* h_idata = (float*)malloc(size);
    float* h_odata = (float*)malloc(size);

    if (!h_idata || !h_odata) {
        fprintf(stderr, "主机内存分配失败。\n");
        return EXIT_FAILURE;
    }

    // 初始化输入矩阵为随机值
    srand(time(NULL));
    for (int i = 0; i < M * N; ++i) {
        h_idata[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 分配设备内存
    float *d_idata, *d_odata;
    CHECK_CUDA_CALL(cudaMalloc(&d_idata, size));
    CHECK_CUDA_CALL(cudaMalloc(&d_odata, size));

    // 将输入数据拷贝到设备
    CHECK_CUDA_CALL(cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice));

    // 定义 CUDA 事件
    cudaEvent_t start, stop;
    CHECK_CUDA_CALL(cudaEventCreate(&start));
    CHECK_CUDA_CALL(cudaEventCreate(&stop));

    // 测试 v0
    CHECK_CUDA_CALL(cudaEventCreate(&start));
    CHECK_CUDA_CALL(cudaEventCreate(&stop));

    CHECK_CUDA_CALL(cudaEventRecord(start, 0));
    mat_transpose_v0(d_idata, d_odata, M, N);
    CHECK_CUDA_CALL(cudaEventRecord(stop, 0));
    CHECK_CUDA_CALL(cudaEventSynchronize(stop));

    float elapsed_ms_v0 = 0.0f;
    CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsed_ms_v0, start, stop));
    float bandwidth_v0 = 2 * size / (elapsed_ms_v0 * 1e6); // GB/s
    float utilization_v0 = (bandwidth_v0 / get_theoretical_bandwidth()) * 100;

    CHECK_CUDA_CALL(cudaEventDestroy(start));
    CHECK_CUDA_CALL(cudaEventDestroy(stop));

    // 测试 v1
    CHECK_CUDA_CALL(cudaEventCreate(&start));
    CHECK_CUDA_CALL(cudaEventCreate(&stop));

    CHECK_CUDA_CALL(cudaEventRecord(start, 0));
    mat_transpose_v1(d_idata, d_odata, M, N);
    CHECK_CUDA_CALL(cudaEventRecord(stop, 0));
    CHECK_CUDA_CALL(cudaEventSynchronize(stop));

    float elapsed_ms_v1 = 0.0f;
    CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsed_ms_v1, start, stop));
    float bandwidth_v1 = 2 * size / (elapsed_ms_v1 * 1e6); // GB/s
    float utilization_v1 = (bandwidth_v1 / get_theoretical_bandwidth()) * 100;

    CHECK_CUDA_CALL(cudaEventDestroy(start));
    CHECK_CUDA_CALL(cudaEventDestroy(stop));


    // 打印结果
    printf("===== 矩阵转置测试结果 =====\n");
    printf("版本 v0:\n");
    printf("用时：%.2f 微秒\n", elapsed_ms_v0 * 1000);
    printf("内存带宽：%.2f GB/s\n", bandwidth_v0);
    printf("带宽利用率：%.2f%%\n", utilization_v0);

    printf("\n版本 v1:\n");
    printf("用时：%.2f 微秒\n", elapsed_ms_v1 * 1000);
    printf("内存带宽：%.2f GB/s\n", bandwidth_v1);
    printf("带宽利用率：%.2f%%\n", utilization_v1);

    // 清理资源
    CHECK_CUDA_CALL(cudaEventDestroy(start));
    CHECK_CUDA_CALL(cudaEventDestroy(stop));
    free(h_idata);
    free(h_odata);
    CHECK_CUDA_CALL(cudaFree(d_idata));
    CHECK_CUDA_CALL(cudaFree(d_odata));

    return EXIT_SUCCESS;
}