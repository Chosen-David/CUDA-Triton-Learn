#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "fast_gemv.cuh"
#include "utility.cuh"
#include "gemv.h"


__global__ void gemv_kstages(const half* __restrict__ A,
                             const half* __restrict__ B,
                             float* __restrict__ C_acc, int N, int K) {
  constexpr int kTileK = gemv_kstages_common::kTileK;
  constexpr int kStages = gemv_kstages_common::kStages;
  constexpr int kPackElems = gemv_kstages_common::kPackElems;
  constexpr int kWarpSize = gemv_kstages_common::kWarpSize;

  extern __shared__ half shared_mem[];
  half* stage_buffers[kStages];
#pragma unroll
  for (int s = 0; s < kStages; ++s) {
    stage_buffers[s] = shared_mem + s * kTileK;
  }

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  if (warps_per_block == 0) {
    return;
  }

  const int total_row_tiles =
      gemv_kstages_common::ceil_div(N, warps_per_block);
  if (total_row_tiles == 0) {
    return;
  }

  auto load_stage = [&](int stage_slot, int tile_idx) {
    const int k_base = tile_idx * kTileK;
    if (k_base >= K) {
      return;
    }

    int tile_len = K - k_base;
    tile_len = tile_len > kTileK ? kTileK : tile_len;

    half* dst = stage_buffers[stage_slot];
    const half* src = A + k_base;

    const int full_packs = tile_len / kPackElems;
    for (int pack = threadIdx.x; pack < full_packs; pack += blockDim.x) {
      reinterpret_cast<uint4*>(dst)[pack] =
          reinterpret_cast<const uint4*>(src)[pack];
    }

    const int tail = tile_len - full_packs * kPackElems;
    const int tail_base = full_packs * kPackElems;
    for (int t = threadIdx.x; t < tail; t += blockDim.x) {
      dst[tail_base + t] = src[tail_base + t];
    }
  };

  const bool warp_active = warp_id < warps_per_block;

  for (int row_tile = blockIdx.x; row_tile < total_row_tiles;
       row_tile += gridDim.x) {
    const int row_idx = row_tile * warps_per_block + warp_id;
    const bool row_valid = warp_active && (row_idx < N);
    const half* b_row =
        row_valid ? B + static_cast<size_t>(row_idx) * K : nullptr;

    float accum = 0.0f;

    const int k_tiles = gemv_kstages_common::ceil_div(K, kTileK);
    const int prefetched = k_tiles < kStages ? k_tiles : kStages;
    for (int s = 0; s < prefetched; ++s) {
      load_stage(s, s);
    }
    __syncthreads();

    int next_tile = prefetched;
    for (int tile = 0; tile < k_tiles; ++tile) {
      const int stage_slot = tile % kStages;
      const int k_base = tile * kTileK;

      int tile_len = K - k_base;
      tile_len = tile_len > kTileK ? kTileK : tile_len;

      const half* a_tile = stage_buffers[stage_slot];
      if (row_valid) {
        const half* b_tile = b_row + k_base;
        const int full_packs = tile_len / kPackElems;
        for (int pack = lane; pack < full_packs; pack += kWarpSize) {
          const uint4 a_vec = reinterpret_cast<const uint4*>(a_tile)[pack];
          const uint4 b_vec = reinterpret_cast<const uint4*>(b_tile)[pack];
          accum += gemv_kstages_common::dot_pack8(a_vec, b_vec);
        }

        const int tail = tile_len - full_packs * kPackElems;
        const int tail_base = full_packs * kPackElems;
        if (tail) {
          for (int t = lane; t < tail; t += kWarpSize) {
            const half a_val = a_tile[tail_base + t];
            const half b_val = b_tile[tail_base + t];
            accum =
                fmaf(__half2float(a_val), __half2float(b_val), accum);
          }
        }
      }

      __syncthreads();
      if (next_tile < k_tiles) {
        load_stage(stage_slot, next_tile);
      }
      ++next_tile;
      __syncthreads();
    }

    const float sum = gemv_kstages_common::warp_reduce_sum(accum);
    if (row_valid && lane == 0) {
      C_acc[row_idx] = sum;
    }
  }
}

///////////////////////////// NORMAL //////////////////////////////
// thread_per_block = blockDim.x
// blockDim.y <= SHARED_MEM_MAX_ROWS
__global__ void gemv_fp16(half* mat, half* vec, half* res, unsigned int n,
                          unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (n >> 3) + j];
      const half2* vec_h1 = (half2*)&vec_val.x;
      const half2* vec_h2 = (half2*)&vec_val.y;
      const half2* vec_h3 = (half2*)&vec_val.z;
      const half2* vec_h4 = (half2*)&vec_val.w;
      const half2* mat_h1 = (half2*)&mat_val.x;
      const half2* mat_h2 = (half2*)&mat_val.y;
      const half2* mat_h3 = (half2*)&mat_val.z;
      const half2* mat_h4 = (half2*)&mat_val.w;
      sum += static_cast<float>(vec_h1->x) * static_cast<float>(mat_h1->x);
      sum += static_cast<float>(vec_h1->y) * static_cast<float>(mat_h1->y);
      sum += static_cast<float>(vec_h2->x) * static_cast<float>(mat_h2->x);
      sum += static_cast<float>(vec_h2->y) * static_cast<float>(mat_h2->y);
      sum += static_cast<float>(vec_h3->x) * static_cast<float>(mat_h3->x);
      sum += static_cast<float>(vec_h3->y) * static_cast<float>(mat_h3->y);
      sum += static_cast<float>(vec_h4->x) * static_cast<float>(mat_h4->x);
      sum += static_cast<float>(vec_h4->y) * static_cast<float>(mat_h4->y);
    }
  }

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}

///////////////////////////// QUANTIZED-INT8 //////////////////////////////

__global__ void gemv_quantized_int8(int8_t* mat, half* vec, half* res,
                                    unsigned int n, half scale, half zero_point,
                                    unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  half4* mat4 = reinterpret_cast<half4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

  float zero_point_f = static_cast<float>(zero_point);
  float scale_f = static_cast<float>(scale);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      half4 mat_val = mat4[row * (n >> 3) + j];
      const half2* vec_h1 = (half2*)&vec_val.x;
      const half2* vec_h2 = (half2*)&vec_val.y;
      const half2* vec_h3 = (half2*)&vec_val.z;
      const half2* vec_h4 = (half2*)&vec_val.w;
      const int8_2* mat_h1 = (int8_2*)&mat_val.x;
      const int8_2* mat_h2 = (int8_2*)&mat_val.y;
      const int8_2* mat_h3 = (int8_2*)&mat_val.z;
      const int8_2* mat_h4 = (int8_2*)&mat_val.w;
      sum += static_cast<float>(vec_h1->x) *
             (static_cast<float>(mat_h1->x) - zero_point_f);
      sum += static_cast<float>(vec_h1->y) *
             (static_cast<float>(mat_h1->y) - zero_point_f);
      sum += static_cast<float>(vec_h2->x) *
             (static_cast<float>(mat_h2->x) - zero_point_f);
      sum += static_cast<float>(vec_h2->y) *
             (static_cast<float>(mat_h2->y) - zero_point_f);
      sum += static_cast<float>(vec_h3->x) *
             (static_cast<float>(mat_h3->x) - zero_point_f);
      sum += static_cast<float>(vec_h3->y) *
             (static_cast<float>(mat_h3->y) - zero_point_f);
      sum += static_cast<float>(vec_h4->x) *
             (static_cast<float>(mat_h4->x) - zero_point_f);
      sum += static_cast<float>(vec_h4->y) *
             (static_cast<float>(mat_h4->y) - zero_point_f);
    }
  }

  sum *= scale_f;

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}

///////////////////////////// QUANTIZED-INT4 //////////////////////////////

// based on previous experiments, num_per_thread can >= 16
__global__ void gemv_quantized_int4(uint4_2* mat, half* vec, half* res,
                                    unsigned int n, half scale, half zero_point,
                                    unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  uint4_2_4* mat4 = reinterpret_cast<uint4_2_4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

  float zero_point_f = static_cast<float>(zero_point);
  float scale_f = static_cast<float>(scale);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 4; iter++) {
    unsigned int j = 2 * (start_idx + iter * blockDim.x);
    if (j < n >> 3) {
      float4 vec_val_1 = vec4[j];  // 8 half
      float4 vec_val_2 = vec4[j + 1];
      const half2* vec_h1 = (half2*)&vec_val_1.x;
      const half2* vec_h2 = (half2*)&vec_val_1.y;
      const half2* vec_h3 = (half2*)&vec_val_1.z;
      const half2* vec_h4 = (half2*)&vec_val_1.w;
      const half2* vec_h5 = (half2*)&vec_val_2.x;
      const half2* vec_h6 = (half2*)&vec_val_2.y;
      const half2* vec_h7 = (half2*)&vec_val_2.z;
      const half2* vec_h8 = (half2*)&vec_val_2.w;

      uint4_2_4 mat_val_1 = mat4[row * (n >> 3) + j];
      uint4_2_4 mat_val_2 = mat4[row * (n >> 3) + j + 1];
      const uint4_2* mat_h1 = (uint4_2*)&mat_val_1.x;
      const uint4_2* mat_h2 = (uint4_2*)&mat_val_1.y;
      const uint4_2* mat_h3 = (uint4_2*)&mat_val_1.z;
      const uint4_2* mat_h4 = (uint4_2*)&mat_val_1.w;
      const uint4_2* mat_h5 = (uint4_2*)&mat_val_2.x;
      const uint4_2* mat_h6 = (uint4_2*)&mat_val_2.y;
      const uint4_2* mat_h7 = (uint4_2*)&mat_val_2.z;
      const uint4_2* mat_h8 = (uint4_2*)&mat_val_2.w;

      sum += static_cast<float>(vec_h1->x) *
             (static_cast<float>(mat_h1->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h1->y) *
             (static_cast<float>(mat_h1->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h2->x) *
             (static_cast<float>(mat_h2->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h2->y) *
             (static_cast<float>(mat_h2->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h3->x) *
             (static_cast<float>(mat_h3->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h3->y) *
             (static_cast<float>(mat_h3->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h4->x) *
             (static_cast<float>(mat_h4->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h4->y) *
             (static_cast<float>(mat_h4->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h5->x) *
             (static_cast<float>(mat_h5->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h5->y) *
             (static_cast<float>(mat_h5->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h6->x) *
             (static_cast<float>(mat_h6->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h6->y) *
             (static_cast<float>(mat_h6->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h7->x) *
             (static_cast<float>(mat_h7->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h7->y) *
             (static_cast<float>(mat_h7->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h8->x) *
             (static_cast<float>(mat_h8->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h8->y) *
             (static_cast<float>(mat_h8->getY()) - zero_point_f);
    }
  }

  sum *= scale_f;

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}

///////////////////////////// REDUCE SUM //////////////////////////////

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
