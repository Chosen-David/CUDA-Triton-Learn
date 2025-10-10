#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "gemv.h"

struct GemvTask {
    const half* __restrict__ A;
    const half* __restrict__ B;
    float* __restrict__ C_acc;
    int N;
    int K;
};

extern "C" __global__
void gemv_kstages_persistent(
    const GemvTask* __restrict__ tasks,
    int num_tasks,
    int* __restrict__ task_counter
) {
    constexpr int kTileK = gemv_kstages_common::kTileK;
    constexpr int kStages = gemv_kstages_common::kStages;
    constexpr int kPackElems = gemv_kstages_common::kPackElems;
    constexpr int kWarpSize = gemv_kstages_common::kWarpSize;

    if (!tasks || !task_counter || num_tasks <= 0) {
        return;
    }

    extern __shared__ half gemv_kstages_persistent_shared_mem[];
    half* stage_buffers[kStages];
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
        stage_buffers[s] = gemv_kstages_persistent_shared_mem + s * kTileK;
    }

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    if (warps_per_block == 0) {
        return;
    }

    const bool warp_active = warp_id < warps_per_block;
    const int threads = blockDim.x;
    __shared__ int current_task;
    __shared__ GemvTask block_task;

    while (true) {
        if (threadIdx.x == 0) {
            current_task = atomicAdd(task_counter, 1);
        }
        __syncthreads();

        const int task_idx = current_task;
        if (task_idx >= num_tasks) {
            break;
        }

        if (threadIdx.x == 0) {
            block_task = tasks[task_idx];
        }
        __syncthreads();

        const half* __restrict__ A = block_task.A;
        const half* __restrict__ B = block_task.B;
        float* __restrict__ C_acc = block_task.C_acc;
        const int N = block_task.N;
        const int K = block_task.K;

        if (!A || !B || !C_acc || N <= 0 || K <= 0) {
            __syncthreads();
            continue;
        }

        const int total_row_tiles =
            gemv_kstages_common::ceil_div(N, warps_per_block);
        if (total_row_tiles == 0) {
            __syncthreads();
            continue;
        }

        const int k_tiles =
            gemv_kstages_common::ceil_div(K, kTileK);
        if (k_tiles == 0) {
            __syncthreads();
            continue;
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
            for (int pack = threadIdx.x; pack < full_packs; pack += threads) {
                reinterpret_cast<uint4*>(dst)[pack] =
                    reinterpret_cast<const uint4*>(src)[pack];
            }

            const int tail = tile_len - full_packs * kPackElems;
            const int tail_base = full_packs * kPackElems;
            for (int t = threadIdx.x; t < tail; t += threads) {
                dst[tail_base + t] = src[tail_base + t];
            }
        };

        const int prefetched = k_tiles < kStages ? k_tiles : kStages;
        for (int row_tile = 0; row_tile < total_row_tiles; ++row_tile) {
            const int row_idx = row_tile * warps_per_block + warp_id;
            const bool row_valid = warp_active && (row_idx < N);
            const half* b_row = row_valid ? B + static_cast<size_t>(row_idx) * K : nullptr;

            float accum = 0.0f;

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
                            accum = fmaf(__half2float(a_val), __half2float(b_val), accum);
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

            __syncthreads();
        }

        __syncthreads();
    }
}

extern "C" __global__
void gemv_kstages_pipeline(
    const half* __restrict__ A,    // [K]
    const half* __restrict__ B,    // [N, K]
    float* __restrict__ C_acc,     // [N] 中间累加（float）
    int N, int K
) {
    constexpr int kTileK = gemv_kstages_common::kTileK;
    constexpr int kStages = gemv_kstages_common::kStages;
    constexpr int kPackElems = gemv_kstages_common::kPackElems;
    constexpr int kWarpSize = gemv_kstages_common::kWarpSize;

    extern __shared__ half gemv_kstages_pipeline_shared_mem[];
    half* stage_buffers[kStages];
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
        stage_buffers[s] = gemv_kstages_pipeline_shared_mem + s * kTileK;
    }

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    if (warps_per_block == 0) {
        return;
    }

    const bool warp_active = warp_id < warps_per_block;
    const int total_row_tiles =
        gemv_kstages_common::ceil_div(N, warps_per_block);

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
            reinterpret_cast<uint4*>(dst)[pack] = reinterpret_cast<const uint4*>(src)[pack];
        }

        const int tail = tile_len - full_packs * kPackElems;
        const int tail_base = full_packs * kPackElems;
        for (int t = threadIdx.x; t < tail; t += blockDim.x) {
            dst[tail_base + t] = src[tail_base + t];
        }
    };

    for (int row_tile = blockIdx.x; row_tile < total_row_tiles; row_tile += gridDim.x) {
        const int row_idx = row_tile * warps_per_block + warp_id;
        const bool row_valid = warp_active && (row_idx < N);
        const half* b_row = row_valid ? B + static_cast<size_t>(row_idx) * K : nullptr;

        float accum = 0.0f;

        const int k_tiles =
            gemv_kstages_common::ceil_div(K, kTileK);
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
                        accum = fmaf(__half2float(a_val), __half2float(b_val), accum);
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
