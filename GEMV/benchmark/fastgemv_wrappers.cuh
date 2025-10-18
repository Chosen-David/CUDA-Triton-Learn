// GEMV/benchmark/fastgemv_wrappers.cuh
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

#include "../gemv.h"          // launch_gemv_with_smem_max
#include "../gemv_smem.cuh"   // gemv_smem kernel
#include "../gemv_smem_cost_model.cuh"  // gemv_smem_detail::compute_num_per_thread_half
#include "fast_gemv.cuh"      // gemv_fp16
#include "../include/smem_log.hpp"  // smemlog::query_or_search

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

namespace fastgemv_impl {
constexpr unsigned int kBlockDimX = 32;
constexpr unsigned int kBlockDimY = 4;

__host__ __device__ inline unsigned int ceil_div(unsigned int v, unsigned int d) {
  return (v + d - 1) / d;
}
__host__ __device__ inline unsigned int compute_num_per_thread(size_t K) {
  const unsigned int n_f4 = static_cast<unsigned int>(K >> 3);
  if (n_f4 == 0) return 8;
  unsigned int f4_per_thread = ceil_div(n_f4, kBlockDimX);
  if (f4_per_thread == 0u) f4_per_thread = 1u;
  return f4_per_thread * 8u;
}
} // namespace fastgemv_impl

// 来自其他 .cu
extern void threadSmem(const half*, const half*, half*, size_t, size_t, cudaStream_t);
extern void warp1Smem(const half*, const half*, half*, size_t, size_t, cudaStream_t);
extern void warp2Smem(const half*, const half*, half*, size_t, size_t, cudaStream_t);
extern void warp4Smem(const half*, const half*, half*, size_t, size_t, cudaStream_t);
extern void warp8Smem(const half*, const half*, half*, size_t, size_t, cudaStream_t);
extern void warp16Smem(const half*, const half*, half*, size_t, size_t, cudaStream_t);

// k-stages Acc 缓存（由 test 主程序分配）
namespace { inline float* g_kstages_accumulator_ext = nullptr; }
inline void set_kstages_acc_buffer(float* ptr) { g_kstages_accumulator_ext = ptr; }

// ============ Wrappers ============
inline void fp16Vec4Wrapper(const half* A, const half* B, half* C,
                            size_t M, size_t K, cudaStream_t stream) {
  if (M == 0 || K == 0) return;
#ifndef NDEBUG
  if ((K & 7) != 0) { std::cerr << "[fp16Vec4Wrapper] K%8!=0\n"; std::exit(EXIT_FAILURE); }
#endif
  using namespace fastgemv_impl;
  const size_t rows_per_tile = kBlockDimY;
  const size_t fast_rows     = (M / rows_per_tile) * rows_per_tile;
  const unsigned int npt     = compute_num_per_thread(K);

  if (fast_rows) {
    dim3 block(kBlockDimX, kBlockDimY);
    dim3 grid(1, static_cast<unsigned int>(fast_rows / rows_per_tile));
    gemv_fp16<<<grid, block, 0, stream>>>(const_cast<half*>(B),
                                          const_cast<half*>(A),
                                          C,
                                          static_cast<unsigned int>(K),
                                          npt);
    CUDA_CHECK(cudaGetLastError());
  }
  if (fast_rows < M) {
    const size_t tail_rows = M - fast_rows;
    warp16Smem(A, B + fast_rows * K, C + fast_rows, tail_rows, K, stream);
    CUDA_CHECK(cudaGetLastError());
  }
}

inline void smemVec4Wrapper(const half* A, const half* B, half* C,
                            size_t M, size_t K, cudaStream_t stream) {

  dim3 block(32, 1);
  dim3 grid(1, 4);

  gemv_smem<<<grid, block>>>(
      const_cast<half*>(B), const_cast<half*>(A), C,
      K, M);
  CUDA_CHECK(cudaGetLastError());
}



inline void fastgemv_default(const half* A, const half* B, half* C,
                             size_t M, size_t K, cudaStream_t stream) {
  if (M == 0 || K == 0) return;
#ifndef NDEBUG
  if ((K & 7) != 0) { std::cerr << "[fastgemv] K%8!=0\n"; std::exit(EXIT_FAILURE); }
#endif
  using namespace fastgemv_impl;
  const size_t rows_per_tile = kBlockDimY;
  const size_t fast_rows     = (M / rows_per_tile) * rows_per_tile;
  const unsigned int npt     = compute_num_per_thread(K);

  if (fast_rows != 0) {
    dim3 block(kBlockDimX, kBlockDimY);
    dim3 grid(1, static_cast<unsigned int>(fast_rows / rows_per_tile));
    launch_gemv_with_smem_max(const_cast<half*>(B), const_cast<half*>(A), C,
                              static_cast<unsigned int>(fast_rows),
                              static_cast<unsigned int>(K),
                              grid, block);
    CUDA_CHECK(cudaGetLastError());
  }
  if (fast_rows < M) {
    const size_t tail_rows = M - fast_rows;
    warp16Smem(A, B + fast_rows * K, C + fast_rows, tail_rows, K, stream);
    CUDA_CHECK(cudaGetLastError());
  }
}

// --- k-stages baseline ---
inline void gemvKStagesWrapper(const half* A, const half* B, half* C,
                               size_t M, size_t K, cudaStream_t stream) {
  (void)C;
  if (M == 0 || K == 0) return;

  if (!g_kstages_accumulator_ext) {
    std::cerr << "kstages accumulator buffer is not initialized.\n";
    std::exit(EXIT_FAILURE);
  }

  constexpr int kThreads = 128;
  constexpr int kWarpSize = 32;
  constexpr int kStages = 2;
  constexpr int kTileK = 1024;

  const int warps_per_block = kThreads / kWarpSize;
  if (warps_per_block == 0) return;

  const unsigned int total_tiles =
      fastgemv_impl::ceil_div(static_cast<unsigned int>(M),
                              static_cast<unsigned int>(warps_per_block));
  unsigned int grid_x = std::max(1u, total_tiles);
  grid_x = std::min(grid_x, 65535u);

  const size_t shared_bytes =
      static_cast<size_t>(kStages) * kTileK * sizeof(half);

  gemv_kstages<<<grid_x, kThreads, shared_bytes, stream>>>(
      A, B, g_kstages_accumulator_ext, static_cast<int>(M),
      static_cast<int>(K));
  CUDA_CHECK(cudaGetLastError());
}
