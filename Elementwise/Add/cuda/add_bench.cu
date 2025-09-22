// Simple benchmark driver for CUDA elementwise add kernels in add.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string>

// Kernels from add.cu
__global__ void add_vec4(float* __restrict__, float* __restrict__, float* __restrict__, int);
__global__ void add_vec2_f16(const half* __restrict__, const half* __restrict__, half* __restrict__, int);
__global__ void add_vec8_f16(const half* __restrict__, const half* __restrict__, half* __restrict__, int);

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(1); \
  } \
} while (0)

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

int main(int argc, char** argv) {
  // Args: dtype[f32|f16] kernel[f32:vec4, f16:vec2|vec8] N iterations
  std::string dtype = argc > 1 ? argv[1] : std::string("f16");
  std::string kname = argc > 2 ? argv[2] : std::string("vec8");
  int N = argc > 3 ? std::atoi(argv[3]) : (1 << 24); // ~16M elements
  int iters = argc > 4 ? std::atoi(argv[4]) : 50;

  int device = 0;
  CUDA_CHECK(cudaSetDevice(device));

  const int block = 256;

  if (dtype == "f32") {
    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, N * sizeof(float)));

    // init by cudaMemset to avoid host copies (not important for profiling)
    CUDA_CHECK(cudaMemset(a, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(b, 0, N * sizeof(float)));

    const int V = 4;
    int tiles = ceil_div(N, V);
    int grid  = ceil_div(tiles, block);

    // warmup
    add_vec4<<<grid, block>>>(a, b, c, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t s, e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int it = 0; it < iters; ++it) add_vec4<<<grid, block>>>(a, b, c, N);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); ms /= iters;
    double bytes = 12.0 * N; // 2 reads + 1 write
    double gbs = (bytes * 1e-9) / (ms * 1e-3);
    printf("[f32/vec4] N=%d grid=%d block=%d avg=%.3fms BW=%.2f GB/s\n", N, grid, block, ms, gbs);

    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(c));
    return 0;
  }

  // f16
  half *a, *b, *c;
  CUDA_CHECK(cudaMalloc(&a, N * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&b, N * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&c, N * sizeof(half)));
  CUDA_CHECK(cudaMemset(a, 0, N * sizeof(half)));
  CUDA_CHECK(cudaMemset(b, 0, N * sizeof(half)));

  int grid = 0;
  if (kname == "vec2") {
    const int V = 2;
    int tiles = ceil_div(N, V);
    grid = ceil_div(tiles, block);
    add_vec2_f16<<<grid, block>>>(a, b, c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int it = 0; it < iters; ++it) add_vec2_f16<<<grid, block>>>(a, b, c, N);
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); ms /= iters;
    double bytes = 6.0 * N; // 2 reads + 1 write (half)
    double gbs = (bytes * 1e-9) / (ms * 1e-3);
    printf("[f16/vec2] N=%d grid=%d block=%d avg=%.3fms BW=%.2f GB/s\n", N, grid, block, ms, gbs);
  } else { // vec8 default
    const int V = 8;
    int tiles = ceil_div(N, V);
    grid = ceil_div(tiles, block);
    add_vec8_f16<<<grid, block>>>(a, b, c, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int it = 0; it < iters; ++it) add_vec8_f16<<<grid, block>>>(a, b, c, N);
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); ms /= iters;
    double bytes = 6.0 * N; // half
    double gbs = (bytes * 1e-9) / (ms * 1e-3);
    printf("[f16/vec8] N=%d grid=%d block=%d avg=%.3fms BW=%.2f GB/s\n", N, grid, block, ms, gbs);
  }

  CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(c));
  return 0;
}

