# CUDA-Triton-Learn

记录自己的CUDA和Triton的学习过程，我会尽量用CUDA和Triton实现各个算子

[CSDN](https://blog.csdn.net/qq_71640350/category_12936188.html)会进行同步

目前：

Elementwise系列：add f16与f32的向量化访存、cute版本

Reduce系列：naive版本、Unroll版本

GEMM系列：naive版本、双缓冲优化

flash_atten系列：更新ing

Sigmoid系列：naive 、f32_vec4

VLLM算子系列:Merge Attention

量化系列： 更新ing

# Ref
1. [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
2. [Triton Doc](https://triton-lang.org/main/getting-started/tutorials/)
3. [Awesome-CUDA-and-HPC](https://github.com/coderonion/awesome-cuda-and-hpc)
