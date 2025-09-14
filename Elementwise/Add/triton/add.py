# 官方文档: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py
# 了解triton: https://isamu-website.medium.com/understanding-the-triton-tutorials-part-1-6191b59ba4c
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid=tl.program_id(axis=0)
    block_start=pid*BLOCK_SIZE;
    offset=block_start+tl.arange(0,BLOCK_SIZE)
    mask=offset<n_elements
    x=tl.load(x_ptr,mask=mask)
    y=tl.load(y_ptr,mask=mask)
    output=x+y
    output_ptr=tl.store(output_ptr+offset,output,mask)
