#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef SHARED_MEM_MAX_ROWS
#define SHARED_MEM_MAX_ROWS 64
#endif

struct GemvSmemTrialTrace {
  unsigned int tile_k_half = 0;
  unsigned int k_stage = 1;
  unsigned int block_x = WARP_SIZE;
  unsigned int block_y = 4;
  double time_ms = std::numeric_limits<double>::infinity();
  double gflops = 0.0;
  size_t shared_bytes = 0;
};

struct GemvSmemCostModelResult {
  unsigned int tile_k_half = 0;
  unsigned int k_stage = 2;
  unsigned int block_x = WARP_SIZE;
  unsigned int block_y = 4;
  double best_time_ms = std::numeric_limits<double>::infinity();
  double best_gflops = 0.0;
  std::vector<GemvSmemTrialTrace> trials;
};

namespace gemv_smem_detail {

inline constexpr unsigned int kTileAlignment = 8u;

inline unsigned int align_to_tile_granularity(unsigned int value) {
  if (value == 0) return kTileAlignment;
  return (value + kTileAlignment - 1u) & ~(kTileAlignment - 1u);
}

inline void check_cuda(cudaError_t err, const char* call) {
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("[gemv_smem_cost_model] ") + call + " failed: " +
        cudaGetErrorString(err));
  }
}

inline unsigned int compute_num_per_thread_half(unsigned int tile_k_half,
                                                unsigned int threads_x) {
  if (threads_x == 0) {
    throw std::invalid_argument("[gemv_smem_cost_model] threads_x must be > 0");
  }
  const unsigned int aligned_tile = align_to_tile_granularity(tile_k_half);
  unsigned int per_thread = (aligned_tile + threads_x - 1u) / threads_x;
  if (per_thread < kTileAlignment) per_thread = kTileAlignment;
  return align_to_tile_granularity(per_thread);
}

inline std::vector<unsigned int> make_256_step_candidates(unsigned int max_tile_half,
                                                          unsigned int K) {
  std::vector<unsigned int> candidates;
  const unsigned int upper = std::min<unsigned int>({max_tile_half, 4096u, K});
  if (upper >= 256u) {
    for (unsigned int tile = 256u; tile <= upper; tile += 256u) {
      candidates.push_back(tile);
    }
  }
  if (candidates.empty() && max_tile_half >= kTileAlignment) {
    candidates.push_back(align_to_tile_granularity(std::min(max_tile_half, K)));
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
  return candidates;
}

}  // namespace gemv_smem_detail

extern "C" __global__
void gemv_smem(half* __restrict__ mat,
               half* __restrict__ vec,
               half* __restrict__ res,
               unsigned int n,
               unsigned int num_per_thread_half,
               unsigned int tile_k_half,
               unsigned int k_stage);

// === 成本模型（随机 half 数据 + tile: 256..4096, stage: 2..4, block 形状搜索） ===
inline GemvSmemCostModelResult gemv_smem_cost_model(
    unsigned int M, unsigned int K,
    cudaStream_t stream = nullptr,
    unsigned int warmup_iters = 5u,
    unsigned int timing_iters = 10u,
    std::vector<unsigned int> candidate_tiles = {}) {

  if ((K & 7u) != 0u) {
    throw std::invalid_argument("[gemv_smem_cost_model] K must be multiple of 8.");
  }
  if (M == 0u) {
    throw std::invalid_argument("[gemv_smem_cost_model] M must be > 0.");
  }
  if (timing_iters == 0u) {
    throw std::invalid_argument("[gemv_smem_cost_model] timing_iters must be > 0.");
  }

  int device = 0;
  gemv_smem_detail::check_cuda(cudaGetDevice(&device), "cudaGetDevice");

  int shared_bytes_default = 0;
  gemv_smem_detail::check_cuda(
      cudaDeviceGetAttribute(&shared_bytes_default,
                             cudaDevAttrMaxSharedMemoryPerBlock, device),
      "cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlock)");

  cudaFuncAttributes func_attr{};
  gemv_smem_detail::check_cuda(
      cudaFuncGetAttributes(&func_attr, gemv_smem),
      "cudaFuncGetAttributes(gemv_smem)");
  const int static_shared_bytes = func_attr.sharedSizeBytes;

  auto align256 = [](int bytes) {
    if (bytes <= 0) return 0;
    return (bytes / 256) * 256;
  };

  int max_dynamic_bytes = align256(shared_bytes_default - static_shared_bytes);
  if (max_dynamic_bytes <= 0) {
    throw std::runtime_error("[gemv_smem_cost_model] insufficient shared memory.");
  }

  const unsigned int max_tile_half =
      static_cast<unsigned int>(max_dynamic_bytes / sizeof(half));

  if (candidate_tiles.empty()) {
    candidate_tiles = gemv_smem_detail::make_256_step_candidates(max_tile_half, K);
  } else {
    for (auto &tile : candidate_tiles) {
      tile = std::min(gemv_smem_detail::align_to_tile_granularity(tile),
                      std::min(max_tile_half, K));
    }
    std::sort(candidate_tiles.begin(), candidate_tiles.end());
    candidate_tiles.erase(std::unique(candidate_tiles.begin(), candidate_tiles.end()),
                          candidate_tiles.end());
    if (candidate_tiles.empty()) {
      candidate_tiles = gemv_smem_detail::make_256_step_candidates(max_tile_half, K);
    }
  }

  // ===== 新增：block 形状候选 =====
  // 你可以按需调整这两个列表
  const unsigned int block_x_candidates[] = {32u, 64u};
  const unsigned int block_y_candidates[] = {2u, 4u, 8u};

  const size_t mat_elems = static_cast<size_t>(M) * static_cast<size_t>(K);
  const size_t vec_elems = static_cast<size_t>(K);
  const size_t res_elems = static_cast<size_t>(M);

  half* d_mat = nullptr;
  half* d_vec = nullptr;
  half* d_res = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  auto cleanup = [&]() {
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (d_mat) cudaFree(d_mat);
    if (d_vec) cudaFree(d_vec);
    if (d_res) cudaFree(d_res);
  };

  GemvSmemCostModelResult best;
  std::vector<GemvSmemTrialTrace> trials;

  try {
    gemv_smem_detail::check_cuda(cudaMalloc(&d_mat, mat_elems * sizeof(half)), "cudaMalloc(d_mat)");
    gemv_smem_detail::check_cuda(cudaMalloc(&d_vec, vec_elems * sizeof(half)), "cudaMalloc(d_vec)");
    gemv_smem_detail::check_cuda(cudaMalloc(&d_res, res_elems * sizeof(half)), "cudaMalloc(d_res)");

    // 随机 half 数据（可复现）
    std::vector<half> h_mat(mat_elems), h_vec(vec_elems), h_res(res_elems, __float2half(0.0f));
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : h_mat) x = __float2half(dist(rng));
    for (auto &x : h_vec) x = __float2half(dist(rng));

    gemv_smem_detail::check_cuda(cudaMemcpyAsync(d_mat, h_mat.data(),
        mat_elems * sizeof(half), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(d_mat)");
    gemv_smem_detail::check_cuda(cudaMemcpyAsync(d_vec, h_vec.data(),
        vec_elems * sizeof(half), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(d_vec)");
    gemv_smem_detail::check_cuda(cudaMemcpyAsync(d_res, h_res.data(),
        res_elems * sizeof(half), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(d_res)");
    gemv_smem_detail::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(init)");

    gemv_smem_detail::check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    gemv_smem_detail::check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (unsigned int block_x : block_x_candidates) {
      if (block_x == 0u || (block_x % WARP_SIZE) != 0u) continue; // 32 的倍数
      for (unsigned int block_y : block_y_candidates) {
        if (block_y == 0u || block_y > SHARED_MEM_MAX_ROWS) continue;
        if (static_cast<unsigned long long>(block_x) * block_y > 1024ull) continue; // 线程上限

        const dim3 block(block_x, block_y);
        const unsigned int grid_y = std::max(1u, (M + block_y - 1) / block_y);
        const dim3 grid(1u, grid_y);

        for (unsigned int tile : candidate_tiles) {
          const unsigned int num_per_thread =
              gemv_smem_detail::compute_num_per_thread_half(tile, block_x);
          const size_t bytes_per_stage = static_cast<size_t>(tile) * sizeof(half);
          if (bytes_per_stage == 0) continue;

          // k_stage 只在 {2,3,4} 中搜索
          for (unsigned int stage = 2u; stage <= 4u; ++stage) {
            const size_t shared_bytes = bytes_per_stage * stage;
            if (shared_bytes == 0 || shared_bytes > static_cast<size_t>(max_dynamic_bytes)) continue;

            // warmup
            for (unsigned int w = 0; w < warmup_iters; ++w) {
              gemv_smem<<<grid, block, shared_bytes, stream>>>(
                  d_mat, d_vec, d_res, K, num_per_thread, tile, stage);
            }
            gemv_smem_detail::check_cuda(cudaGetLastError(), "gemv_smem(warmup)");

            // timing
            gemv_smem_detail::check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
            for (unsigned int t = 0; t < timing_iters; ++t) {
              gemv_smem<<<grid, block, shared_bytes, stream>>>(
                  d_mat, d_vec, d_res, K, num_per_thread, tile, stage);
            }
            gemv_smem_detail::check_cuda(cudaGetLastError(), "gemv_smem(timing)");
            gemv_smem_detail::check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
            gemv_smem_detail::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

            float elapsed_ms = 0.0f;
            gemv_smem_detail::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                                         "cudaEventElapsedTime");

            const double avg_ms = elapsed_ms / timing_iters;
            const double flops  = 2.0 * static_cast<double>(M) * static_cast<double>(K);
            const double gflops = flops / (avg_ms * 1.0e6);

            trials.push_back(GemvSmemTrialTrace{
              tile, stage, block_x, block_y, avg_ms, gflops, shared_bytes
            });

            if (gflops > best.best_gflops) {
              best.tile_k_half = tile;
              best.k_stage     = stage;
              best.block_x     = block_x;
              best.block_y     = block_y;
              best.best_gflops = gflops;
              best.best_time_ms = avg_ms;
            }
          }
        }
      }
    }

    gemv_smem_detail::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(finalize)");
  } catch (...) {
    cleanup();
    throw;
  }

  best.trials = std::move(trials);
  cleanup();
  return best;
}

#ifndef TILE_K_HALF
#define TILE_K_HALF 2048u
#endif

__inline__ __device__ float warpReduceSum(float val, int width=WARP_SIZE) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ unsigned int swizzle_half2_idx(
    unsigned int j_idx,
    unsigned int word_idx,
    unsigned int full_blocks,
    unsigned int tail_h2_base,
    unsigned int tail_guard) {
  unsigned int block = j_idx >> 3;
  if (block > full_blocks) block = full_blocks;
  const unsigned int lane_in_block = j_idx - (block << 3);
  if (tail_guard != 0u && block == full_blocks) {
    return tail_h2_base + (lane_in_block << 2) + word_idx;
  }
  return (block << 5) + ((word_idx << 3) | lane_in_block);
}

#if !defined(GEMV_SMEM_ONLY_DECL)
// ---------------- 修复后的 Kernel（正确的装载流水线） ----------------
extern "C" __global__
void gemv_smem(half* __restrict__ mat,   // [M x K], row-major
               half* __restrict__ vec,   // [K]
               half* __restrict__ res,   // [M]
               unsigned int n,           // K
               unsigned int num_per_thread_half,
               unsigned int tile_k_half,
               unsigned int k_stage) {
  const unsigned int tid  = threadIdx.x;
  const unsigned int row  = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int lane = tid & (WARP_SIZE - 1);
  const unsigned int warp = tid >> 5;

  const unsigned int n_f4_total = n >> 3;

  extern __shared__ half smem_raw[];
  half2* __restrict__ smem_h2 = reinterpret_cast<half2*>(smem_raw);
  const unsigned int stages = (k_stage == 0u) ? 1u : k_stage;
  const unsigned int stage_stride_h2 = tile_k_half >> 1;
  if (stage_stride_h2 == 0u) return;

  float4* __restrict__ mat4 = reinterpret_cast<float4*>(mat);
  float4* __restrict__ vec4 = reinterpret_cast<float4*>(vec);

  float sum = 0.0f;

  const unsigned int tile_f4 = tile_k_half >> 3;         // 每 tile 的 float4 数
  const unsigned int npt_f4  = num_per_thread_half >> 3; // 每线程处理的 float4 数
  if (tile_f4 == 0u || npt_f4 == 0u) return;

  const unsigned int total_tiles = (n_f4_total + tile_f4 - 1u) / tile_f4;

  // 预取前 stages 个 tile 到 smem
  unsigned int load_idx = 0u;
  const unsigned int initial_prefetch = min(stages, total_tiles);
  for (; load_idx < initial_prefetch; ++load_idx) {
    const unsigned int base_f4 = load_idx * tile_f4;
    const unsigned int cur_f4  = min(tile_f4, n_f4_total - base_f4);
    const unsigned int full_blocks = cur_f4 >> 3;
    const unsigned int tail_guard  = cur_f4 & 0x7u;
    const unsigned int tail_h2_base = full_blocks << 5;
    half2* stage_base = smem_h2 + (load_idx % stages) * stage_stride_h2;

#pragma unroll
    for (unsigned int it = 0; it < npt_f4; ++it) {
      unsigned int j = tid + it * blockDim.x;
      if (j < cur_f4) {
        const float4 v = vec4[base_f4 + j];
        const half2* v_h2 = reinterpret_cast<const half2*>(&v.x);
#pragma unroll
        for (unsigned int word = 0; word < 4u; ++word) {
          const unsigned int sm_idx = swizzle_half2_idx(
              j, word, full_blocks, tail_h2_base, tail_guard);
          stage_base[sm_idx] = v_h2[word];
        }
      }
    }
    __syncthreads();
  }

  unsigned int compute_idx = 0u;

  while (compute_idx < total_tiles) {
    const unsigned int stage = compute_idx % stages;
    const unsigned int base_f4 = compute_idx * tile_f4;
    const unsigned int cur_f4  = min(tile_f4, n_f4_total - base_f4);
    const unsigned int full_blocks = cur_f4 >> 3;
    const unsigned int tail_guard  = cur_f4 & 0x7u;
    const unsigned int tail_h2_base = full_blocks << 5;
    half2* stage_base = smem_h2 + stage * stage_stride_h2;

    // 计算当前 stage
#pragma unroll
    for (unsigned int it = 0; it < npt_f4; ++it) {
      unsigned int j = tid + it * blockDim.x;
      if (j < cur_f4) {
        half2 vec_vals[4];
#pragma unroll
        for (unsigned int word = 0; word < 4u; ++word) {
          const unsigned int sm_idx = swizzle_half2_idx(
              j, word, full_blocks, tail_h2_base, tail_guard);
          vec_vals[word] = stage_base[sm_idx];
        }

        float4 a = mat4[row * n_f4_total + (base_f4 + j)];
        const half2* a1 = reinterpret_cast<const half2*>(&a.x);
        const half2* a2 = reinterpret_cast<const half2*>(&a.y);
        const half2* a3 = reinterpret_cast<const half2*>(&a.z);
        const half2* a4 = reinterpret_cast<const half2*>(&a.w);

        sum += __half2float(vec_vals[0].x) * __half2float(a1->x);
        sum += __half2float(vec_vals[0].y) * __half2float(a1->y);
        sum += __half2float(vec_vals[1].x) * __half2float(a2->x);
        sum += __half2float(vec_vals[1].y) * __half2float(a2->y);
        sum += __half2float(vec_vals[2].x) * __half2float(a3->x);
        sum += __half2float(vec_vals[2].y) * __half2float(a3->y);
        sum += __half2float(vec_vals[3].x) * __half2float(a4->x);
        sum += __half2float(vec_vals[3].y) * __half2float(a4->y);
      }
    }
    __syncthreads();

    // 用刚释放的 stage 去加载下一块
    if (load_idx < total_tiles) {
      const unsigned int load_stage = stage;
      const unsigned int load_base_f4 = load_idx * tile_f4;
      const unsigned int load_cur_f4  = min(tile_f4, n_f4_total - load_base_f4);
      const unsigned int load_full_blocks = load_cur_f4 >> 3;
      const unsigned int load_tail_guard  = load_cur_f4 & 0x7u;
      const unsigned int load_tail_h2_base = load_full_blocks << 5;
      half2* load_stage_base = smem_h2 + load_stage * stage_stride_h2;

#pragma unroll
      for (unsigned int it = 0; it < npt_f4; ++it) {
        unsigned int j = tid + it * blockDim.x;
        if (j < load_cur_f4) {
          const float4 v = vec4[load_base_f4 + j];
          const half2* v_h2 = reinterpret_cast<const half2*>(&v.x);
#pragma unroll
          for (unsigned int word = 0; word < 4u; ++word) {
            const unsigned int sm_idx = swizzle_half2_idx(
                j, word, load_full_blocks, load_tail_h2_base, load_tail_guard);
            load_stage_base[sm_idx] = v_h2[word];
          }
        }
      }
      __syncthreads();
      ++load_idx;
    }

    ++compute_idx;
  }

  // 归约
  sum = warpReduceSum(sum, WARP_SIZE);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) res[row] = __float2half(sum);
    return;
  }

  __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  if (lane == 0) warpLevelSums[threadIdx.y][warp] = sum;
  __syncthreads();

  const int numWarps = blockDim.x / WARP_SIZE;
  float acc = 0.0f;
  if (warp == 0) {
    acc = (lane < numWarps) ? warpLevelSums[threadIdx.y][lane] : 0.0f;
    acc = warpReduceSum(acc, WARP_SIZE);
    if (lane == 0) res[row] = __float2half(acc);
  }
}
#endif  // !defined(GEMV_SMEM_ONLY_DECL)

// 便捷启动器（保持原样；注意：这里仍使用编译时固定的 TILE_K_HALF 与传入 k_stage）
inline void launch_gemv_with_smem_max(
    half* d_mat, half* d_vec, half* d_res,
    unsigned int M, unsigned int K,
    dim3 grid, dim3 block,
    unsigned int num_per_thread_half,
    cudaStream_t stream = 0,
    unsigned int k_stage = 2u) {

  if (num_per_thread_half & 7u) {
    num_per_thread_half = (num_per_thread_half + 7u) & ~7u;
  }
  if (k_stage == 0u) k_stage = 1u;

  const size_t shared_bytes =
      static_cast<size_t>(TILE_K_HALF) * static_cast<size_t>(k_stage) * sizeof(half);

  gemv_smem<<<grid, block, shared_bytes, stream>>>(
      d_mat, d_vec, d_res, K, num_per_thread_half, TILE_K_HALF, k_stage);
}
