# ref: 
# 知乎：https://zhuanlan.zhihu.com/p/1899562146477609112
# 代码：https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/openai-triton/fused-softmax/triton_fused_softmax.py
import torch
import triton
import triton.language as tl

def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements; 读取MN元素；写M个元素
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements; 读取MN+M元素；写入MN元素
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements; 读取MN元素；写入MN元素
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements; 读取MN元素；写M个元素
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements; 读取MN M元素；写入MN元素
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements;
    return ret # 共：读取5MN+2M元素；写了3MN+2M个元素


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    k_stages: tl.constexpr,
):
    # 起始位置
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    # 从 row_start 开始，以 row_step 为间隔跳着处理多行
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages = k_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0,BLOCK_SIZE) # 生成一个从 0 到 BLOCK_SIZE - 1 的整型向量（也就是一系列列偏移索引）
        input_ptrs = row_start_ptr + col_offsets;
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs ,mask=mask, other=-float('inf'))
        row_safe = row - tl.max(row, axis=0)
        numerator = tl.exp(row_safe) # 这个是 element-wise 操作，因此不需要指定axis
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def get_device_properties(device_id=None):
    import pycuda.driver as cuda

    device = (
        cuda.Device(device_id)
        if device_id is not None
        else torch.cuda.current_device()
    )
    NUM_SM = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
    NUM_REGS = device.get_attribute(
        cuda.device_attribute.MAX_REGISTERS_PER_BLOCK
    )
    SIZE_SMEM = device.get_attribute(
        cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
    )
    WARP_SIZE = device.get_attribute(cuda.device_attribute.WARP_SIZE)
    return NUM_SM, NUM_REGS, SIZE_SMEM, WARP_SIZE

DEVICE = torch.cuda.current_device()
NUM_SM, NUM_REGS, SIZE_SMEM, WARP_SIZE = get_device_properties(DEVICE)
print(
    f"NUM_SM: {NUM_SM}, NUM_REGS: {NUM_REGS}, "
    f"SIZE_SMEM: {SIZE_SMEM}, WARP_SIZE: {WARP_SIZE}"
)


def get_num_programs(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    k_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(x)
    kernel = softmax_kernel.warmup(
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        k_stages=k_stages,
        num_warps=num_warps,
        grid=(1,),
    )
    kernel._init_handles() # 建立句柄，初始化这个kernel的相关信息
    n_regs = kernel.n_regs # 每个线程要用多少寄存器
    size_smem = kernel.metadata.shared 
    # 计算 在资源限制下，一个 SM（Streaming Multiprocessor）上能容纳多少个 block
    occupancy = NUM_REGS // (n_regs * num_warps * WARP_SIZE) 
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    return num_programs

NUM_PROGRAMS = get_num_programs(torch.randn(4096, 2048, device="cuda"))


def triton_softmax(x: torch.Tensor):
    """Compute row-wise softmax of X using Triton"""
    n_rows, n_cols = x.shape
    # The block size of each loop iteration is the smallest power of
    # two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    # Number of software pipelining stages.
    k_stages = 4 if SIZE_SMEM > 200000 else 2
    # Allocate output
    y = torch.empty_like(x)
    num_programs = min(NUM_PROGRAMS, n_rows)

    # Create a number of persistent programs.
    softmax_kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        k_stages=k_stages,
        num_warps=num_warps,
    )
    return y


torch.manual_seed(0)
x = torch.randn(1823, 781, device="cuda")
y_triton = triton_softmax(x)
y_torch = naive_softmax(x)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],  # argument names to use as an x-axis for the plot
        x_vals=[
            256 * i for i in range(1, 64)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "triton-fused-softmax",
            "torch-fused-softmax",
            "torch-naive-softmax",
        ],  # possible values for `line_arg``
        line_names=[
            "Triton Fused Softmax",
            "Torch Fused Softmax",
            "Torch Naive Softmax",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        xlabel=f"M, {torch.cuda.get_device_name(DEVICE)}",  # label name for the x-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={
            "N": 2048
        },  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == "torch-naive-softmax":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    if provider == "triton-fused-softmax":
        ms = triton.testing.do_bench(lambda: triton_softmax(x))
    if provider == "torch-fused-softmax":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True, save_path="./")
    





    