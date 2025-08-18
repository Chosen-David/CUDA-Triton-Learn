#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>           // 定义 half / __half2 运算
#include <cute/tensor.hpp>       // 来自 CUTLASS 3：-I<cutlass>/include
#include <../cute/add_vec_cute.cu>

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(1); \
  } \
} while (0)


// --------------------------- CPU reference ---------------------------
static inline float h2f(half h) { return __half2float(h); }
static inline half  f2h(float f) { return __float2half(f); }

void host_axpbyc(std::vector<half>& z,
                 const std::vector<half>& x,
                 const std::vector<half>& y,
                 half a, half b, half c) {
  int n = (int)z.size();
  float af = h2f(a), bf = h2f(b), cf = h2f(c);
  for (int i = 0; i < n; ++i) {
    float xi = h2f(x[i]);
    float yi = h2f(y[i]);
    float zi = af * xi + bf * yi + cf;
    z[i] = f2h(zi);
  }
}

// --------------------------- Utility ---------------------------
template <typename T>
static inline T ceil_div(T a, T b) { return (a + b - 1) / b; }

// --------------------------- Main test ---------------------------
int main(int argc, char** argv) {
  // 测试规模（可自定义）
  int num = (1 << 20) + 7;         // 刻意做成非整除，验证尾部谓词
  if (argc > 1) num = std::atoi(argv[1]);

  const int kEPT = 8;              // 每线程元素数，与 kernel 模板参数一致
  const int block = 256;
  const int tiles = ceil_div(num, kEPT);
  const int grid  = ceil_div(tiles, block);

  printf("[Conf] num=%d, kEPT=%d, grid=%d, block=%d\n", num, kEPT, grid, block);

  // 标量
  float af = 1.2345f, bf = -0.5f, cf = 0.125f;
  half  a = f2h(af), b = f2h(bf), c = f2h(cf);

  // Host 数据
  std::vector<half> hx(num), hy(num), hz(num), href(num);
  for (int i = 0; i < num; ++i) {
    float vx = std::sin(0.001f * i);
    float vy = std::cos(0.001f * i);
    hx[i] = f2h(vx);
    hy[i] = f2h(vy);
  }

  // Device 内存
  half *dx = nullptr, *dy = nullptr, *dz = nullptr;
  CUDA_CHECK(cudaMalloc(&dx, num * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dy, num * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dz, num * sizeof(half)));

  CUDA_CHECK(cudaMemcpy(dx, hx.data(), num * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dy, hy.data(), num * sizeof(half), cudaMemcpyHostToDevice));

  // 预热
  add_vec_cute<kEPT><<<grid, block>>>(dz, num, dx, dy, a, b, c);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 计时（多次迭代取均值）
  const int iters = 100;
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  for (int it = 0; it < iters; ++it) {
    add_vec_cute<kEPT><<<grid, block>>>(dz, num, dx, dy, a, b, c);
  }
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
  ms /= iters;  // 平均每次 kernel 的毫秒

  // 拷回 & 校验
  CUDA_CHECK(cudaMemcpy(hz.data(), dz, num * sizeof(half), cudaMemcpyDeviceToHost));
  host_axpbyc(href, hx, hy, a, b, c);

  int    mismatch = 0;
  double max_abs_err = 0.0, mean_abs_err = 0.0;
  for (int i = 0; i < num; ++i) {
    float zf   = h2f(hz[i]);
    float zref = h2f(href[i]);
    float err  = std::fabs(zf - zref);
    max_abs_err   = std::max(max_abs_err, (double)err);
    mean_abs_err += err;
    if (err > 2e-2f) { // 半精度阈值，按需调整
      if (++mismatch <= 5) {
        printf("mismatch @%d: got=%g ref=%g (abs err=%g)\n", i, zf, zref, err);
      }
    }
  }
  mean_abs_err /= num;

  // 粗略性能估算
  // 访存：读 x/y（各2B），写 z（2B） => 6 * num 字节
  // 计算：每元素做 2 mul + 2 add ≈ 4 FLOP（把 a*x 和 b*y+c 看成两次 FMA ~ 4 FLOP）
  double bytes = 6.0 * num;
  double gbs = (bytes * 1e-9) / (ms * 1e-3);
  double gflops = (4.0 * num * 1e-9) / (ms * 1e-3);

  printf("[Time]  avg %.3f ms  |  BW ~ %.2f GB/s  |  ~ %.2f GFLOP/s\n", ms, gbs, gflops);
  printf("[Check] mismatch<=5 shown | max_abs_err=%.4g | mean_abs_err=%.4g\n",
         max_abs_err, mean_abs_err);

  // 清理
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  CUDA_CHECK(cudaFree(dz));

  return 0;
}
