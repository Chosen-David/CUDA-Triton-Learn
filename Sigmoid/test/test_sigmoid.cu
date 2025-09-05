// test_sigmoid.cu  -- self-contained, standalone
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../sigmoid.cu"

// ---------------------------- 配置与兜底宏 ----------------------------
#ifndef CHECK_CUDA
#define CHECK_CUDA(expr) do {                                   \
  cudaError_t _err = (expr);                                     \
  if (_err != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
            cudaGetErrorString(_err), __FILE__, __LINE__);       \
    std::exit(1);                                                \
  }                                                              \
} while(0)
#endif






// ---------------------------- 计时工具（host） ----------------------------
static float elapsed_ms(cudaEvent_t beg, cudaEvent_t end) {
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, beg, end));
  return ms;
}

// 关键修正处：用非 const 输入指针，避免模板推断与 kernel 参数不一致
template <typename Kernel, typename T>
static float bench_kernel(Kernel k, dim3 grid, dim3 block,
                          T* dx, T* dy, int N, int iters) {
  cudaEvent_t beg, end;
  CHECK_CUDA(cudaEventCreate(&beg));
  CHECK_CUDA(cudaEventCreate(&end));
  CHECK_CUDA(cudaDeviceSynchronize());      // 干净起点
  CHECK_CUDA(cudaEventRecord(beg));
  for (int i=0;i<iters;++i) k<<<grid, block>>>(dx, dy, N);
  CHECK_CUDA(cudaEventRecord(end));
  CHECK_CUDA(cudaEventSynchronize(end));
  float ms = elapsed_ms(beg, end);
  CHECK_CUDA(cudaEventDestroy(beg));
  CHECK_CUDA(cudaEventDestroy(end));
  return ms / iters;
}

// ---------------------------- 测试主逻辑（host） ----------------------------
void test_sigmoid(int N = (1<<20), int iters = 200) {
  // 1) 生成输入
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<float> hx(N);
  for (int i=0;i<N;++i) hx[i] = dist(rng);

  std::vector<half> hx_half(N);
  for (int i=0;i<N;++i) hx_half[i] = __float2half(hx[i]);

  // 2) 设备内存
  float *dx_f32=nullptr, *dy_f32=nullptr;
  half  *dx_f16=nullptr, *dy_f16=nullptr;
  // 作为“naive 参考”的输出缓冲
  float *dref_f32=nullptr;
  half  *dref_f16=nullptr;

  CHECK_CUDA(cudaMalloc(&dx_f32,   N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dy_f32,   N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dx_f16,   N*sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dy_f16,   N*sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dref_f32, N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dref_f16, N*sizeof(half)));

  CHECK_CUDA(cudaMemcpy(dx_f32, hx.data(),      N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dx_f16, hx_half.data(), N*sizeof(half),  cudaMemcpyHostToDevice));

  // 3) 通用 launch 参数
  dim3 block(256);
  dim3 grid_naive((N + block.x - 1) / block.x);
  dim3 grid_v4(((N/4) + block.x - 1) / block.x);
  dim3 grid_v2(((N/2) + block.x - 1) / block.x);

  // 4) warm-up
  for (int i=0;i<10;++i) {
    sigmoid_f32_naive<<<grid_naive, block>>>(dx_f32, dy_f32, N);
    sigmoid_f16_naive<<<grid_naive, block>>>(dx_f16, dy_f16, N);
    sigmoid_f32_vec4 <<<grid_v4,    block>>>(dx_f32, dy_f32, N);
    sigmoid_f16_vec2 <<<grid_v2,    block>>>(dx_f16, dy_f16, N);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // 5) 生成 naive 基准输出
  sigmoid_f32_naive<<<grid_naive, block>>>(dx_f32, dref_f32, N);
  sigmoid_f16_naive<<<grid_naive, block>>>(dx_f16, dref_f16, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  // 6) 计时
  float t_f32_naive = bench_kernel(sigmoid_f32_naive, grid_naive, block, dx_f32, dy_f32, N, iters);
  float t_f16_naive = bench_kernel(sigmoid_f16_naive, grid_naive, block, dx_f16, dy_f16, N, iters);
  float t_f32_v4    = bench_kernel(sigmoid_f32_vec4,  grid_v4,    block, dx_f32, dy_f32, N, iters);
  float t_f16_v2    = bench_kernel(sigmoid_f16_vec2,  grid_v2,    block, dx_f16, dy_f16, N, iters);

  // 7) 正确性校验（与 naive 对比）
  // 为避免被测输出被覆盖，额外各跑一次被测 kernel
  sigmoid_f32_vec4<<<grid_v4, block>>>(dx_f32, dy_f32, N);
  sigmoid_f16_vec2<<<grid_v2, block>>>(dx_f16, dy_f16, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> y_f32(N), ref_f32(N);
  std::vector<half>  y_f16(N), ref_f16(N);
  CHECK_CUDA(cudaMemcpy(y_f32.data(),   dy_f32,   N*sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ref_f32.data(), dref_f32, N*sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(y_f16.data(),   dy_f16,   N*sizeof(half),  cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ref_f16.data(), dref_f16, N*sizeof(half),  cudaMemcpyDeviceToHost));

  double max_abs_err_f32_vs_naive = 0.0;
  for (int i=0;i<N;++i) {
    max_abs_err_f32_vs_naive = std::max(
      max_abs_err_f32_vs_naive,
      std::abs((double)y_f32[i] - (double)ref_f32[i])
    );
  }

  double max_abs_err_f16_vs_naive = 0.0;
  for (int i=0;i<N;++i) {
    float yf   = __half2float(y_f16[i]);
    float yref = __half2float(ref_f16[i]);
    max_abs_err_f16_vs_naive = std::max(
      max_abs_err_f16_vs_naive,
      std::abs((double)yf - (double)yref)
    );
  }

  // 8) 打印结果
  printf("N=%d, iters=%d\n", N, iters);
  printf("[Time]  f32_naive : %.3f ms\n", t_f32_naive);
  printf("[Time]  f32_vec4  : %.3f ms\n", t_f32_v4);
  printf("[Time]  f16_naive : %.3f ms\n", t_f16_naive);
  printf("[Time]  f16_vec2  : %.3f ms\n", t_f16_v2);

  printf("[MaxErr] f32_vec4 vs f32_naive : %.3e\n", max_abs_err_f32_vs_naive);
  printf("[MaxErr] f16_vec2 vs f16_naive : %.3e\n", max_abs_err_f16_vs_naive);

  // 9) 清理
  cudaFree(dx_f32); cudaFree(dy_f32);
  cudaFree(dx_f16); cudaFree(dy_f16);
  cudaFree(dref_f32); cudaFree(dref_f16);
}

#ifdef BUILD_STANDALONE
int main() {
  test_sigmoid(); // N = 1<<20, iters = 200
  return 0;
}
#endif