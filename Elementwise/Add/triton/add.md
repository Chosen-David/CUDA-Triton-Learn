# Ref
1. [官方示例](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py)
2. [学习代码](https://github.com/Chosen-David/CUDA-Triton-Learn)

# Add

## 导入库和指定device
```py
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
```
获取当前Pytorch所处的设备作为device

## Triton的运作方式

注意与CUDA不同，triton指定的是pid，一个program进行一批数据的处理。虽然它也定义了BLOCK_SIZE但是貌似不完全等价CUDA的BLOCK里面线程数，你在这里可以理解为BLOCK_SIZE就是一个pid处理的元素的数量。

真正的线程数由 launch 侧的 num_warps 决定：每个 program 使用 num_warps × 32 个线程协同执行。

```py
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

pid 是 program的 id；每个 program 处理 BLOCK_SIZE 个元素；tl.arange(0, BLOCK_SIZE) 生成这 BLOCK_SIZE 个元素在块内的相对索引；与 block_start 相加得到的 offsets 是该 instance 要处理的全局元素位置。(这些并不是“每个线程的 tid”，而是同一 instance 内并行计算的元素索引向量)

我认为这样的好处是你只需要以pid为单位进行操作了，而不需要向下划分warp和tid的操作细节，在这里统一抽象成了pid。

```py
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
```
这里创建了mask之后内部会自动实现pid内部每个元素处理的并行(自动判断是否被mask住然后load进来)

```py
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```
后面的运算和相加也很顺理成章了。


## TODO::一个pid可以处理更多的任务吗

经常写CUDA的同学知道，我们的CUDA经常指定一个BLOCK_SIZE的block后让它内部每个tid处理不止一个元素，那么Triton也可以这样吗？


## TODO::关于num_warps、num_stages

Triton 的 triton.Config 给出了默认：num_stages=3（同时默认 num_warps=4）。但是如果希望更深入的理解Triton还是有必要关注一下。






