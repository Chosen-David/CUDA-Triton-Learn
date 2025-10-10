// GEMV/test/test_gemv.cu — 精简主程序
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <random>
#include <vector>
#include "../include/fastgemv_config.hpp"
#include "../include/smem_log.hpp"

#include "../gemv_smem.cuh"      // gemv_smem_detail::compute_num_per_thread_half
#include "../gemv.h"
#include "../gemv_kstages.cuh"
#include "../gemv_kstages_persistance.cuh"

#include "../benchmark/benchmark_types.hpp"
#include "../benchmark/benchmark_io.hpp"
#include "../benchmark/fastgemv_wrappers.cuh"
#include "../benchmark/fast_gemv.cuh"   // gemv_fp16 kernel decl

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err__ = (expr);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::cerr << "CUDA error " << cudaGetErrorString(err__) << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                 \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)
#endif

// sweep 配置
static const std::vector<size_t> k_test_rows = []{
  std::vector<size_t> v; for (int i=1;i<=10;++i) v.push_back(2048u*i); return v;
}();
static const std::vector<size_t> k_test_dimensions = []{
  std::vector<size_t> v; for (size_t k=2048;k<=50000;k+=2048) if ((k&7)==0) v.push_back(k); return v;
}();

static void fill_random(std::vector<half>& data, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : data) v = __float2half(dist(rng));
}

int main() {
  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

  const auto& m_values = k_test_rows;
  const auto& k_values = k_test_dimensions;
  const size_t max_M = *std::max_element(m_values.begin(), m_values.end());
  const size_t max_K = *std::max_element(k_values.begin(), k_values.end());
  const size_t max_B_elems = max_M * max_K;

  std::vector<half> host_A(max_K), host_B(max_B_elems);
  std::mt19937 rng(123);
  fill_random(host_A, rng); fill_random(host_B, rng);

  half *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, max_K * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB, max_B_elems * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dC, max_M * sizeof(half)));
  float* dC_acc=nullptr; CUDA_CHECK(cudaMalloc(&dC_acc, max_M * sizeof(float)));
  set_kstages_acc_buffer(dC_acc);

  const KernelSpec kernels[] = {
    {"fast_gemv",  fp16Vec4Wrapper,   OutputKind::kHalf},
    {"smem_vec4",  smemVec4Wrapper,   OutputKind::kHalf},
    {"thread_smem",threadSmem,        OutputKind::kHalf},
    {"warp1_smem", warp1Smem,         OutputKind::kHalf},
    {"warp2_smem", warp2Smem,         OutputKind::kHalf},
    {"warp4_smem", warp4Smem,         OutputKind::kHalf},
    {"warp8_smem", warp8Smem,         OutputKind::kHalf},
    {"warp16_smem",warp16Smem,        OutputKind::kHalf},
    {"kstages",    gemvKStagesWrapper,OutputKind::kFloat},
  };
  const char* baseline_kernel = "fast_gemv";

  std::vector<float> host_reference(max_M), gpu_output_float(max_M);
  std::vector<half>  gpu_output_half(max_M);
  std::vector<float> host_A_float(max_K);
  std::vector<BenchmarkResult> results;
  results.reserve(m_values.size()*k_values.size()*(sizeof(kernels)/sizeof(kernels[0])));

  constexpr int warmup_iters = 4, measure_iters = 10;

  std::cout << "GEMV benchmark (C = A * B, A is 1xK, B is MxK)\n";

  for (size_t M : m_values) {
    std::cout << "\nM=" << M << "\n";
    for (size_t K : k_values) {
#ifndef NDEBUG
      if ((K & 7) != 0) { std::cerr << "[main] K%8!=0\n"; std::exit(EXIT_FAILURE); }
#endif
      CUDA_CHECK(cudaMemcpyAsync(dA, host_A.data(), K*sizeof(half),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(dB, host_B.data(), (M*K)*sizeof(half),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemsetAsync(dC, 0, M * sizeof(half), stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      for (size_t i=0;i<K;++i) host_A_float[i]=__half2float(host_A[i]);
      for (size_t row=0; row<M; ++row) {
        const half* b_row = host_B.data() + row*K;
        float acc=0.0f;
        for (size_t col=0; col<K; ++col) acc += host_A_float[col]*__half2float(b_row[col]);
        host_reference[row]=acc;
      }

      std::cout << "  K=" << K << "\n";
      bool baseline_checked = false;

      for (const auto& kernel : kernels) {
        for (int i=0;i<warmup_iters;++i) kernel.fn(dA, dB, dC, M, K, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaEvent_t start{}, stop{};
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i=0;i<measure_iters;++i) kernel.fn(dA, dB, dC, M, K, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

        const double avg_ms = elapsed_ms / double(measure_iters);
        const double gflops = (2.0*double(M)*double(K)) / (avg_ms*1.0e6);

        std::cout << "    " << kernel.name << ": " << avg_ms << " ms, "
                  << gflops << " GFLOP/s\n";
        results.push_back(BenchmarkResult{M, K, kernel.name, avg_ms, gflops});

        if (!baseline_checked && std::strcmp(kernel.name, baseline_kernel)==0) {
          if (kernel.output_kind == OutputKind::kHalf) {
            CUDA_CHECK(cudaMemcpyAsync(gpu_output_half.data(), dC, M*sizeof(half),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (size_t r=0;r<M;++r) gpu_output_float[r]=__half2float(gpu_output_half[r]);
          } else {
            CUDA_CHECK(cudaMemcpyAsync(gpu_output_float.data(), dC_acc, M*sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
          }
          auto tol = [&](OutputKind kind, size_t Kdim){
            struct { double atol, rtol, l2; } t{};
            if (kind==OutputKind::kFloat) { t.atol=1e-3; t.rtol=std::min(2e-2, 8e-4+2e-6*double(Kdim)); t.l2=0.015; }
            else { t.atol=5e-3; t.rtol=std::min(5e-2, 2e-3+4e-6*double(Kdim)); t.l2=0.02; }
            return t;
          }(kernel.output_kind, K);

          double max_abs=0.0, max_rel=0.0; long double num=0.0L, den=0.0L;
          for (size_t r=0;r<M;++r) {
            const double ref = host_reference[r], got = gpu_output_float[r];
            const double diff = std::fabs(ref-got); max_abs = std::max(max_abs, diff);
            const double denom = std::fabs(ref);
            const double rel = denom>0 ? diff/denom : diff; max_rel = std::max(max_rel, rel);
            num += (got-ref)*(got-ref); den += ref*ref;
          }
          const double rel_l2 = (den>0) ? std::sqrt(double(num/den)) : std::sqrt(double(num));
          const bool ok_elem = (max_abs<=tol.atol) || (max_rel<=tol.rtol);
          const bool ok_l2   = (rel_l2<=tol.l2);

          if (!(ok_elem || ok_l2)) {
            std::cerr << "Verification failed for baseline " << kernel.name
                      << " at M="<<M<<", K="<<K
                      << " (max abs err="<<max_abs
                      << ", max rel err="<<max_rel
                      << ", rel_l2="<<rel_l2<<")\n";
            std::exit(EXIT_FAILURE);
          }
          std::cout << "      baseline verification ok "
                    << "(max abs "<<max_abs<<", max rel "<<max_rel
                    << ", rel_l2 "<<rel_l2<<")\n";
          baseline_checked = true;
        }
      }
    }
  }

  // 输出与画图
  benchio::write_csv("gemv_performance.csv", results);
  if (benchio::emit_plot_script("plot_gemv_performance.py"))
    benchio::run_plot_script("plot_gemv_performance.py");

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dC_acc));
  return 0;
}
