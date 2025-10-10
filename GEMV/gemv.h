#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define WARP_SIZE 32

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err__ = (expr);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::cerr << "CUDA error " << cudaGetErrorString(err__) << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                 \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

  
namespace gemv_kstages_common {

constexpr int kTileK = 1024;
constexpr int kStages = 2;
constexpr int kPackElems = 8;
constexpr int kWarpSize = 32;

__device__ __forceinline__ int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  unsigned mask = __activemask();
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}


__device__ __forceinline__ float warpReduceSum(float sum,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}

__device__ __forceinline__ float dot_pack8(uint4 a_vec, uint4 b_vec) {
  const half* a_half = reinterpret_cast<const half*>(&a_vec);
  const half* b_half = reinterpret_cast<const half*>(&b_vec);
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sum = fmaf(__half2float(a_half[i]), __half2float(b_half[i]), sum);
  }
  return sum;
}

}  // namespace gemv_kstages_common

