#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "fast_gemv.cuh"

#define gemv_kstages gemv_kstages_staged
#define shared_mem gemv_kstages_staged_shared_mem
#include "../gemv_kstages.cuh"
#undef shared_mem
#undef gemv_kstages

struct BenchmarkResult;

#ifdef __linux__
#define PERFORMANCE_PYTHON "python3"
#else
#define PERFORMANCE_PYTHON "python"
#endif

static void generate_performance_plot(const std::vector<BenchmarkResult>& results,
                                      const char* csv_path,
                                      const char* png_path) {
  if (results.empty()) {
    return;
  }

  const char* script_path = "multi_gemv_performance_plot_tmp.py";
  {
    std::ofstream script(script_path);
    if (!script) {
      std::cerr << "Failed to open " << script_path
                << " for writing performance plot script." << std::endl;
      return;
    }
    script << "import csv\n"
           << "import sys\n"
           << "from collections import defaultdict\n"
           << "import matplotlib\n"
           << "matplotlib.use('Agg')\n"
           << "import matplotlib.pyplot as plt\n"
           << "csv_path, png_path = sys.argv[1], sys.argv[2]\n"
           << "data = defaultdict(list)\n"
           << "with open(csv_path, 'r', newline='') as f:\n"
           << "    reader = csv.DictReader(f)\n"
           << "    for row in reader:\n"
           << "        try:\n"
           << "            k = int(row['K'])\n"
           << "            gflops = float(row['gflops'])\n"
           << "        except (KeyError, ValueError, TypeError):\n"
           << "            continue\n"
           << "        kernel = row.get('kernel', 'unknown')\n"
           << "        data[kernel].append((k, gflops))\n"
           << "plt.figure(figsize=(10, 6))\n"
           << "for kernel in sorted(data.keys()):\n"
           << "    series = sorted(data[kernel])\n"
           << "    if not series:\n"
           << "        continue\n"
           << "    ks, gfs = zip(*series)\n"
           << "    plt.plot(ks, gfs, marker='o', linewidth=1.8, label=kernel)\n"
           << "plt.xlabel('K')\n"
           << "plt.ylabel('GFLOP/s')\n"
           << "plt.title('Multi-GEMV Performance')\n"
           << "plt.grid(True, linestyle='--', alpha=0.3)\n"
           << "plt.legend(loc='best', frameon=False)\n"
           << "plt.tight_layout()\n"
           << "plt.savefig(png_path, dpi=200)\n"
           << "plt.close()\n";
  }

  std::string command =
      std::string(PERFORMANCE_PYTHON) + " \"" + script_path + "\" \"" +
      csv_path + "\" \"" + png_path + "\"";
  const int ret = std::system(command.c_str());
  if (ret != 0) {
    std::cerr << "Failed to generate performance plot (exit code "
              << ret << ")." << std::endl;
  }
  std::remove(script_path);
}

#undef PERFORMANCE_PYTHON
#define gemv_kstages gemv_kstages_pipeline
#define gemv_kstages_persistent gemv_kstages_persistent_kernel
#define shared_mem gemv_kstages_pipeline_shared_mem
#include "../gemv_kstages_persistance.cuh"
#undef shared_mem
#undef gemv_kstages
#undef gemv_kstages_persistent

enum class OutputKind { kHalf, kFloat };

namespace {
GemvTask* g_device_tasks = nullptr;
size_t g_device_task_capacity = 0;
int* g_device_task_counter = nullptr;
}  // namespace

static const std::vector<size_t> k_test_dimensions = [] {
  std::vector<size_t> values;
  values.reserve(8);
  size_t k = 256;
  while (k < 28672) {
    values.push_back(k);
    k *= 2;
  }
  if (values.empty() || values.back() != 28672) {
    values.push_back(28672);
  }
  return values;
}();

namespace fastgemv_impl {
constexpr unsigned int kBlockDimX = 32;
constexpr unsigned int kBlockDimY = 4;

inline unsigned int ceil_div(unsigned int value, unsigned int divisor) {
  return (value + divisor - 1) / divisor;
}

inline unsigned int compute_num_per_thread(size_t K) {
  const unsigned int float4_per_row =
      static_cast<unsigned int>((K + 7) / 8);
  if (float4_per_row == 0) {
    return 8;
  }
  return std::max(1u, ceil_div(float4_per_row, kBlockDimX)) * 8u;
}
}  // namespace fastgemv_impl

extern void threadSmem(const half*, const half*, half*, size_t, size_t,
                       cudaStream_t);
extern void warp1Smem(const half*, const half*, half*, size_t, size_t,
                      cudaStream_t);
extern void warp2Smem(const half*, const half*, half*, size_t, size_t,
                      cudaStream_t);
extern void warp4Smem(const half*, const half*, half*, size_t, size_t,
                      cudaStream_t);
extern void warp8Smem(const half*, const half*, half*, size_t, size_t,
                      cudaStream_t);
extern void warp16Smem(const half*, const half*, half*, size_t, size_t,
                       cudaStream_t);

void fastgemv(const half* A, const half* B, half* C, size_t M, size_t K,
              cudaStream_t stream) {
  if (M == 0 || K == 0) {
    return;
  }

  const size_t rows_per_tile = fastgemv_impl::kBlockDimY;
  const size_t fast_rows = (M / rows_per_tile) * rows_per_tile;

  const unsigned int num_per_thread =
      fastgemv_impl::compute_num_per_thread(K);

  if (fast_rows != 0) {
    dim3 block(fastgemv_impl::kBlockDimX, fastgemv_impl::kBlockDimY);
    dim3 grid(1, static_cast<unsigned int>(fast_rows / rows_per_tile));
    gemv_fp16<<<grid, block, 0, stream>>>(const_cast<half*>(B),
                                          const_cast<half*>(A), C,
                                          static_cast<unsigned int>(K),
                                          num_per_thread);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
      std::cerr << "fast_gemv launch failed: "
                << cudaGetErrorString(launch_err) << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  if (fast_rows < M) {
    const size_t tail_rows = M - fast_rows;
    warp16Smem(A, B + fast_rows * K, C + fast_rows, tail_rows, K, stream);
  }
}

using MultiKernelFn = void (*)(const half*, const half*, half*, float*,
                               size_t, size_t, size_t, cudaStream_t);

struct KernelSpec {
  const char* name;
  MultiKernelFn fn;
  OutputKind output_kind;
};

struct BenchmarkResult {
  size_t K;
  const char* kernel_name;
  double avg_ms;
  double gflops;
};

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err__ = (expr);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::cerr << "CUDA error " << cudaGetErrorString(err__) << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                 \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

static void fill_random(std::vector<half>& data, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : data) {
    v = __float2half(dist(rng));
  }
}

static void ensure_persistent_buffers(size_t num_tasks) {
  if (num_tasks == 0) {
    return;
  }
  if (num_tasks > g_device_task_capacity) {
    if (g_device_tasks) {
      CUDA_CHECK(cudaFree(g_device_tasks));
    }
    CUDA_CHECK(cudaMalloc(&g_device_tasks, num_tasks * sizeof(GemvTask)));
    g_device_task_capacity = num_tasks;
  }
  if (!g_device_task_counter) {
    CUDA_CHECK(cudaMalloc(&g_device_task_counter, sizeof(int)));
  }
}

static void fastgemv_multi(const half* A, const half* B, half* C, float*,
                           size_t num_tasks, size_t M, size_t K,
                           cudaStream_t stream) {
  const size_t strideA = K;
  const size_t strideB = M * K;
  const size_t strideC = M;
  for (size_t task = 0; task < num_tasks; ++task) {
    const half* taskA = A + task * strideA;
    const half* taskB = B + task * strideB;
    half* taskC = C + task * strideC;
    fastgemv(taskA, taskB, taskC, M, K, stream);
  }
}

static void thread_smem_multi(const half* A, const half* B, half* C,
                              float*, size_t num_tasks, size_t M, size_t K,
                              cudaStream_t stream) {
  const size_t strideA = K;
  const size_t strideB = M * K;
  const size_t strideC = M;
  for (size_t task = 0; task < num_tasks; ++task) {
    threadSmem(A + task * strideA, B + task * strideB,
               C + task * strideC, M, K, stream);
  }
}

static void warp1_smem_multi(const half* A, const half* B, half* C, float*,
                             size_t num_tasks, size_t M, size_t K,
                             cudaStream_t stream) {
  const size_t strideA = K;
  const size_t strideB = M * K;
  const size_t strideC = M;
  for (size_t task = 0; task < num_tasks; ++task) {
    warp1Smem(A + task * strideA, B + task * strideB,
              C + task * strideC, M, K, stream);
  }
}

static void warp2_smem_multi(const half* A, const half* B, half* C, float*,
                             size_t num_tasks, size_t M, size_t K,
                             cudaStream_t stream) {
  const size_t strideA = K;
  const size_t strideB = M * K;
  const size_t strideC = M;
  for (size_t task = 0; task < num_tasks; ++task) {
    warp2Smem(A + task * strideA, B + task * strideB,
              C + task * strideC, M, K, stream);
  }
}

static void warp4_smem_multi(const half* A, const half* B, half* C, float*,
                             size_t num_tasks, size_t M, size_t K,
                             cudaStream_t stream) {
  const size_t strideA = K;
  const size_t strideB = M * K;
  const size_t strideC = M;
  for (size_t task = 0; task < num_tasks; ++task) {
    warp4Smem(A + task * strideA, B + task * strideB,
              C + task * strideC, M, K, stream);
  }
}

static void warp8_smem_multi(const half* A, const half* B, half* C, float*,
                             size_t num_tasks, size_t M, size_t K,
                             cudaStream_t stream) {
  const size_t strideA = K;
  const size_t strideB = M * K;
  const size_t strideC = M;
  for (size_t task = 0; task < num_tasks; ++task) {
    warp8Smem(A + task * strideA, B + task * strideB,
              C + task * strideC, M, K, stream);
  }
}

static void warp16_smem_multi(const half* A, const half* B, half* C,
                              float*, size_t num_tasks, size_t M, size_t K,
                              cudaStream_t stream) {
  const size_t strideA = K;
  const size_t strideB = M * K;
  const size_t strideC = M;
  for (size_t task = 0; task < num_tasks; ++task) {
    warp16Smem(A + task * strideA, B + task * strideB,
               C + task * strideC, M, K, stream);
  }
}

static void gemvKStagesSingle(const half* A, const half* B, float* C_acc,
                              size_t M, size_t K, cudaStream_t stream) {
  if (M == 0 || K == 0) {
    return;
  }
  if (!C_acc) {
    std::cerr << "kstages accumulator buffer is not initialized."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  constexpr int kThreads = 128;
  constexpr int kWarpSize = 32;
  constexpr int kStages = 2;
  constexpr int kTileK = 1024;
  const int warps_per_block = kThreads / kWarpSize;
  if (warps_per_block == 0) {
    return;
  }
  const unsigned int total_tiles = fastgemv_impl::ceil_div(
      static_cast<unsigned int>(M), static_cast<unsigned int>(warps_per_block));
  unsigned int grid_x = std::max(1u, total_tiles);
  grid_x = std::min(grid_x, 65535u);
  const size_t shared_bytes =
      static_cast<size_t>(kStages) * kTileK * sizeof(half);
  gemv_kstages_staged<<<grid_x, kThreads, shared_bytes, stream>>>(
      A, B, C_acc, static_cast<int>(M), static_cast<int>(K));
  CUDA_CHECK(cudaGetLastError());
}

static void gemv_kstages_multi(const half* A, const half* B, half*,
                               float* C_acc, size_t num_tasks, size_t M,
                               size_t K, cudaStream_t stream) {
  const size_t strideA = K;
  const size_t strideB = M * K;
  const size_t strideC = M;
  for (size_t task = 0; task < num_tasks; ++task) {
    const half* taskA = A + task * strideA;
    const half* taskB = B + task * strideB;
    float* taskC = C_acc + task * strideC;
    gemvKStagesSingle(taskA, taskB, taskC, M, K, stream);
  }
}

static void gemv_kstages_persistent_multi(const half* A, const half* B,
                                          half*, float* C_acc,
                                          size_t num_tasks, size_t M,
                                          size_t K, cudaStream_t stream) {
  if (num_tasks == 0 || M == 0 || K == 0) {
    return;
  }
  ensure_persistent_buffers(num_tasks);
  std::vector<GemvTask> host_tasks(num_tasks);
  const size_t strideA = K;
  const size_t strideB = M * K;
  const size_t strideC = M;
  for (size_t task = 0; task < num_tasks; ++task) {
    host_tasks[task].A = A + task * strideA;
    host_tasks[task].B = B + task * strideB;
    host_tasks[task].C_acc = C_acc + task * strideC;
    host_tasks[task].N = static_cast<int>(M);
    host_tasks[task].K = static_cast<int>(K);
  }

  CUDA_CHECK(cudaMemcpyAsync(
      g_device_tasks, host_tasks.data(),
      num_tasks * sizeof(GemvTask), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(g_device_task_counter, 0, sizeof(int), stream));

  constexpr int kThreads = 128;
  constexpr int kWarpSize = 32;
  constexpr size_t kStages = gemv_kstages_common::kStages;
  constexpr size_t kTileK = gemv_kstages_common::kTileK;
  const unsigned int max_blocks =
      static_cast<unsigned int>(std::min(num_tasks, static_cast<size_t>(65535)));
  unsigned int grid_x = max_blocks;
  grid_x = std::max(1u, grid_x);
  const size_t shared_bytes =
      static_cast<size_t>(kStages) * kTileK * sizeof(half);
  gemv_kstages_persistent_kernel<<<grid_x, kThreads, shared_bytes, stream>>>(
      g_device_tasks, static_cast<int>(num_tasks), g_device_task_counter);
  CUDA_CHECK(cudaGetLastError());
}

int main() {
  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  constexpr size_t M = 2048;
  constexpr size_t kNumTasks = 8;
  const std::vector<size_t>& k_values = k_test_dimensions;
  const size_t max_K =
      *std::max_element(k_values.begin(), k_values.end());

  std::vector<half> host_A_full(kNumTasks * max_K);
  std::vector<half> host_B_full(kNumTasks * M * max_K);
  std::mt19937 rng(321);
  fill_random(host_A_full, rng);
  fill_random(host_B_full, rng);

  half* dA = nullptr;
  half* dB = nullptr;
  half* dC = nullptr;
  float* dC_acc = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, kNumTasks * max_K * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB, kNumTasks * M * max_K * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dC, kNumTasks * M * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dC_acc, kNumTasks * M * sizeof(float)));

  const KernelSpec kernels[] = {
      {"fast_gemv_seq", fastgemv_multi, OutputKind::kHalf},
      {"thread_smem_seq", thread_smem_multi, OutputKind::kHalf},
      {"warp1_smem_seq", warp1_smem_multi, OutputKind::kHalf},
      {"warp2_smem_seq", warp2_smem_multi, OutputKind::kHalf},
      {"warp4_smem_seq", warp4_smem_multi, OutputKind::kHalf},
      {"warp8_smem_seq", warp8_smem_multi, OutputKind::kHalf},
      {"warp16_smem_seq", warp16_smem_multi, OutputKind::kHalf},
      {"kstages_seq", gemv_kstages_multi, OutputKind::kFloat},
      {"kstages_persistent", gemv_kstages_persistent_multi,
       OutputKind::kFloat},
  };

  std::vector<float> gpu_output_float(kNumTasks * M);
  std::vector<half> gpu_output_half(kNumTasks * M);
  std::vector<BenchmarkResult> results;
  results.reserve(k_values.size() *
                  (sizeof(kernels) / sizeof(kernels[0])));

  constexpr int warmup_iters = 3;
  constexpr int measure_iters = 8;

  std::cout << "M=" << M << ", tasks=" << kNumTasks
            << " multi-GEMV benchmark (tasks: C_i = A_i * B_i)\n";

  for (size_t K : k_values) {
    const size_t elemsA = kNumTasks * K;
    const size_t elemsB = kNumTasks * M * K;
    std::vector<half> host_A_slice(elemsA);
    std::vector<half> host_B_slice(elemsB);

    for (size_t task = 0; task < kNumTasks; ++task) {
      const half* srcA = host_A_full.data() + task * max_K;
      half* dstA = host_A_slice.data() + task * K;
      std::copy(srcA, srcA + K, dstA);
      for (size_t row = 0; row < M; ++row) {
        const half* srcB =
            host_B_full.data() + (task * M + row) * max_K;
        half* dstB =
            host_B_slice.data() + (task * M + row) * K;
        std::copy(srcB, srcB + K, dstB);
      }
    }

    CUDA_CHECK(cudaMemcpyAsync(dA, host_A_slice.data(),
                               elemsA * sizeof(half),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dB, host_B_slice.data(),
                               elemsB * sizeof(half),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> host_A_float(elemsA);
    std::vector<float> host_reference(kNumTasks * M);
    for (size_t task = 0; task < kNumTasks; ++task) {
      const half* a_half = host_A_slice.data() + task * K;
      float* a_float = host_A_float.data() + task * K;
      for (size_t col = 0; col < K; ++col) {
        a_float[col] = __half2float(a_half[col]);
      }
      for (size_t row = 0; row < M; ++row) {
        const half* b_row =
            host_B_slice.data() + (task * M + row) * K;
        float acc = 0.0f;
        for (size_t col = 0; col < K; ++col) {
          acc = fmaf(a_float[col], __half2float(b_row[col]), acc);
        }
        host_reference[task * M + row] = acc;
      }
    }

    std::cout << "\nK=" << K << "\n";
    for (const auto& kernel : kernels) {
      CUDA_CHECK(cudaMemsetAsync(dC, 0, kNumTasks * M * sizeof(half),
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(dC_acc, 0,
                                 kNumTasks * M * sizeof(float), stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      for (int i = 0; i < warmup_iters; ++i) {
        kernel.fn(dA, dB, dC, dC_acc, kNumTasks, M, K, stream);
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));

      cudaEvent_t start{};
      cudaEvent_t stop{};
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));
      CUDA_CHECK(cudaEventRecord(start, stream));
      for (int i = 0; i < measure_iters; ++i) {
        kernel.fn(dA, dB, dC, dC_acc, kNumTasks, M, K, stream);
      }
      CUDA_CHECK(cudaEventRecord(stop, stream));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));

      const double avg_ms =
          elapsed_ms / static_cast<double>(measure_iters);
      const double gflops =
          (2.0 * static_cast<double>(kNumTasks) *
           static_cast<double>(M) * static_cast<double>(K)) /
          (avg_ms * 1.0e6);

      std::cout << "  " << kernel.name << ": " << avg_ms << " ms, "
                << gflops << " GFLOP/s" << std::endl;
      results.push_back({K, kernel.name, avg_ms, gflops});

      if (kernel.output_kind == OutputKind::kHalf) {
        CUDA_CHECK(cudaMemcpyAsync(gpu_output_half.data(), dC,
                                   kNumTasks * M * sizeof(half),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (size_t idx = 0; idx < kNumTasks * M; ++idx) {
          gpu_output_float[idx] =
              __half2float(gpu_output_half[idx]);
        }
      } else {
        CUDA_CHECK(cudaMemcpyAsync(gpu_output_float.data(), dC_acc,
                                   kNumTasks * M * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }

      double max_abs_err = 0.0;
      double max_rel_err = 0.0;
      for (size_t idx = 0; idx < kNumTasks * M; ++idx) {
        const double ref = static_cast<double>(host_reference[idx]);
        const double got = static_cast<double>(gpu_output_float[idx]);
        const double diff = std::fabs(ref - got);
        max_abs_err = std::max(max_abs_err, diff);
        const double denom = std::fabs(ref);
        const double rel = denom > 1e-6 ? diff / denom : diff;
        max_rel_err = std::max(max_rel_err, rel);
      }
      const double abs_tol =
          (kernel.output_kind == OutputKind::kFloat) ? 1e-3 : 5e-2;
      const double rel_tol =
          (kernel.output_kind == OutputKind::kFloat) ? 5e-3 : 5e-2;
      if (max_abs_err > abs_tol && max_rel_err > rel_tol) {
        std::cerr << "Verification failed for kernel " << kernel.name
                  << " at K=" << K << " (max abs err=" << max_abs_err
                  << ", max rel err=" << max_rel_err << ")"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::cout << "    verification ok (max abs " << max_abs_err
                << ", max rel " << max_rel_err << ")" << std::endl;
    }
  }

  const char* csv_path = "multi_gemv_performance.csv";
  {
    std::ofstream csv(csv_path);
    if (!csv) {
      std::cerr << "Failed to open " << csv_path << " for writing."
                << std::endl;
    } else {
      csv << "kernel,K,avg_ms,gflops\n";
      for (const auto& entry : results) {
        csv << entry.kernel_name << ',' << entry.K << ','
            << entry.avg_ms << ',' << entry.gflops << '\n';
      }
      generate_performance_plot(
          results, csv_path, "multi_gemv_performance.png");
      std::cout << "\nWrote benchmark data to " << csv_path << std::endl;
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dC_acc));
  if (g_device_tasks) {
    CUDA_CHECK(cudaFree(g_device_tasks));
  }
  if (g_device_task_counter) {
    CUDA_CHECK(cudaFree(g_device_task_counter));
  }

  return 0;
}
