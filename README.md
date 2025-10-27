# CUDA-Triton-Learn

## 概述

记录自己的CUDA和Triton的学习过程，我会尽量用CUDA和Triton实现各个算子

[CSDN](https://blog.csdn.net/qq_71640350/category_12936188.html)会进行同步

## 目前进展

### 基础算子

Elementwise系列：add f16与f32的向量化访存、cute版本、triton版本

Reduce系列：naive版本、Unroll版本

GEMM系列：naive版本、双缓冲优化

GEMV系列：FastGEMV版本、smem版本

### 大模型算子

RoPE系列：triton版本

Sigmoid系列：naive 、f32_vec4

FlashAttention系列：cute版本、cuda版本

PagedAttention系列：cuda版本

MergeAttention：triton版本

量化系列： LLM.int8()版本

# Ref
1. [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
2. [Triton Doc](https://triton-lang.org/main/getting-started/tutorials/)
3. [Awesome-CUDA-and-HPC](https://github.com/coderonion/awesome-cuda-and-hpc)
