#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "cutlass/bfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

// ref:cd CUDA-Triton-Learn/third_party/cutlass/include/cutlass/gemm/kernel/gemm_grouped.h
// ref:cd CUDA-Triton-Learn/third_party/cutlass/examples/24_gemm_grouped/gemm_grouped.cu


