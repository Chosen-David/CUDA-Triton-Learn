#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

//TODO:我希望设置的blockDim.y能够大一点，发挥smem的优势，比如你可以按照256递增到2048得到搜索空间
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

inline bool is_launch_configuration_error(cudaError_t err) {
  switch (err) {
    case cudaSuccess:
      return false;
    case cudaErrorInvalidConfiguration:
    case cudaErrorInvalidValue:
    case cudaErrorLaunchOutOfResources:
#ifdef cudaErrorTooManyResourcesRequested
    case cudaErrorTooManyResourcesRequested:
#endif
      return true;
    default:
      return false;
  }
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

  const unsigned int block_x_candidates[] = {32u, 64u};
  const unsigned int block_y_candidates[] = {4u, 8u, 16u, 32u};

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
      if (block_x == 0u || (block_x % WARP_SIZE) != 0u) continue;
      for (unsigned int block_y : block_y_candidates) {
        if (block_y == 0u || block_y > SHARED_MEM_MAX_ROWS) continue;
        if (static_cast<unsigned long long>(block_x) * block_y > 1024ull) continue;

        const dim3 block(block_x, block_y);
        const unsigned int grid_y = std::max(1u, (M + block_y - 1) / block_y);
        const dim3 grid(1u, grid_y);

        for (unsigned int tile : candidate_tiles) {
          const unsigned int num_per_thread =
              gemv_smem_detail::compute_num_per_thread_half(tile, block_x);
          const size_t bytes_per_stage = static_cast<size_t>(tile) * sizeof(half);
          if (bytes_per_stage == 0) continue;

          for (unsigned int stage = 2u; stage <= 4u; ++stage) {
            const size_t shared_bytes = bytes_per_stage * stage;
            if (shared_bytes == 0 || shared_bytes > static_cast<size_t>(max_dynamic_bytes)) continue;

            bool skip_stage = false;

            for (unsigned int w = 0; w < warmup_iters; ++w) {
              gemv_smem<<<grid, block, shared_bytes, stream>>>(
                  d_mat, d_vec, d_res, K, num_per_thread, tile, stage);
            }
            cudaError_t warmup_err = cudaGetLastError();
            if (warmup_err != cudaSuccess) {
              if (gemv_smem_detail::is_launch_configuration_error(warmup_err)) {
                skip_stage = true;
              } else {
                gemv_smem_detail::check_cuda(warmup_err, "gemv_smem(warmup)");
              }
            }
            if (skip_stage) continue;

            gemv_smem_detail::check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
            for (unsigned int t = 0; t < timing_iters; ++t) {
              gemv_smem<<<grid, block, shared_bytes, stream>>>(
                  d_mat, d_vec, d_res, K, num_per_thread, tile, stage);
            }
            cudaError_t timing_err = cudaGetLastError();
            if (timing_err != cudaSuccess) {
              if (gemv_smem_detail::is_launch_configuration_error(timing_err)) {
                skip_stage = true;
              } else {
                gemv_smem_detail::check_cuda(timing_err, "gemv_smem(timing)");
              }
            }
            if (skip_stage) continue;

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
