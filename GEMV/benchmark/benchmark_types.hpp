// GEMV/benchmark/benchmark_types.hpp
#pragma once
#include <cstddef>

enum class OutputKind { kHalf, kFloat };

struct KernelSpec {
  const char* name;
  void (*fn)(const __half*, const __half*, __half*, size_t, size_t, cudaStream_t);
  OutputKind output_kind;
};

struct BenchmarkResult {
  size_t M;
  size_t K;
  const char* kernel_name;
  double avg_ms;
  double gflops;
};
