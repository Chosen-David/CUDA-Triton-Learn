# Ref
1. [官方示例](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py)
2. 

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





