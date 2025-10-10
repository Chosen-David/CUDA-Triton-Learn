这个benchmark搜集了我截至25/10/9搜集的开源高性能的GEMV的实现

其中：

thread_smem.cu: 每个线程计算 1 个 C 输出，用 shared memory

warp1_smem.cu: 每个 warp 计算 1 个 C 输出，用 shared memory

warp2_smem.cu: 每个 warp 计算 2 个 C 输出 用 shared memory

warp4_smem: 每个 warp 计算 4 个 C 输出

warp8_smem: 每个 warp 计算 8 个输出

warp16_smem: 每个 warp 计算 16 输出

来源于Bruce-Lee-LY 的 [cuda_hgemv](https://github.com/Bruce-Lee-LY/cuda_hgemv) 项目

fast_gemv.cu

来源于 wangsiping97的 [FastGEMV](https://github.com/wangsiping97/FastGEMV) 项目



