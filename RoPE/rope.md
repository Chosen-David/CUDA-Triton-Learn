# Ref
1. [旋转编码](https://www.zhihu.com/tardis/bd/art/647109286)
2. [源码](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py)
3. [关于Sinusoidal函数](https://zhuanlan.zhihu.com/p/359500899)
4. [关于多变量函数的泰勒展开](https://dezeming.top/wp-content/uploads/2021/06/%E5%A4%9A%E5%85%83%E5%87%BD%E6%95%B0%EF%BC%88%E5%8F%8A%E5%90%91%E9%87%8F%E5%87%BD%E6%95%B0%EF%BC%89%E7%9A%84%E6%B3%B0%E5%8B%92%E5%B1%95%E5%BC%80.pdf)


# 源码
主要是对于每个 token、每个 head 的前 ROTARY_DIM 个通道做旋转位置编码（RoPE），结果写到 OUT。
## 函数签名
```py
@triton.jit
def rotary_kernel(...,
    ROTARY_DIM: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
```

## 设置 tile 宽度与半维度
```py
BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
ROTARY_DIM_HALF = ROTARY_DIM // 2
```
向上取整至最近的 2 的幂。如果𝑛已经是 2 的幂（例如 64, 128, 256 等），那么 next_power_of_2(n) = n 本身；否则返回大于𝑛的最小的 2 的幂（例如 n = 100，则返回 128；n = 129 返回 256，以此类推）。

前 ROTARY_DIM 的维度分成若干对（2D rotation），所以处理最前面的 `ROTARY_DIM` 维比后面的维度更关键。可能是希望对齐这`ROTARY_DIM` 维。

## 三维并行网格的“块坐标”
```py
pid_head  = tl.program_id(axis=0)
pid_m     = tl.program_id(axis=1)
pid_batch = tl.program_id(axis=2)
```
指定triton启动的三个Kernel维度:

axis=0 按 head/通道块 切分，

axis=1 按 token 段（每段 BLOCK_M 个 token） 切分，

axis=2 按 batch 切分。

## 根据是否 varlen，定位本实例负责的“批起点”

```py
if not IS_VARLEN:
    X   = X   + pid_batch * stride_x_batch
    OUT = OUT + pid_batch * stride_out_batch
else:
    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen    = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
    X   = X   + start_idx * stride_x_seqlen
    OUT = OUT + start_idx * stride_out_seqlen
```

**定长**：直接用 batch 步幅把指针移到第 pid_batch 个样本的起点；

**变长压紧**：利用 CU_SEQLENS[b]/CU_SEQLENS[b+1] 得到该样本在扁平拼接后的起止位置（start_idx、seqlen），再把指针移到这段的开头。

## 无效 tile 的“早退”
```py
if pid_m * BLOCK_M >= seqlen:
    return
```

## 计算本 tile 的头索引、token 索引与“相对位置”
```py
rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)  # 头块内各 head
rm = pid_m   * BLOCK_M + tl.arange(0, BLOCK_M)   # 该 token 段内各 token

if not IS_SEQLEN_OFFSETS_TENSOR:
    rm_cs = rm + SEQLEN_OFFSETS                   # 标量偏移（续写时整体平移）
else:
    rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)  # 每样本各自的偏移
```
其中rm_cs 是用于查 COS/SIN 的位置索引：把局部 token 索引 rm 加上“序列起点偏移”

## 构造 COS/SIN 的指针并带掩码加载
```py
rk_half = tl.arange(0, BLOCK_K // 2)
COS = COS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
SIN = SIN + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
mask_cs = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)
if CONJUGATE:
    sin = -sin
```

## 非交错（NeoX）路径：前半/后半配对旋转
```py
if not INTERLEAVED:
    # 组装 X/OUT 的三维指针：head×token×headdim(半宽)
    X   = X   + (rh[:,None,None]*stride_x_nheads + rm[None,:,None]*stride_x_seqlen + rk_half[None,None,:]*stride_x_headdim)
    OUT = OUT + (rh[:,None,None]*stride_out_nheads+ rm[None,:,None]*stride_out_seqlen+ rk_half[None,None,:]*stride_out_headdim)
    mask = (rh[:,None,None] < nheads) & (rm[None,:,None] < seqlen) & (rk_half[None,None,:] < ROTARY_DIM_HALF)

    # 前半与后半各 load 一次
    x0 = tl.load(X,                                 mask=mask, other=0.0).to(tl.float32)        # 前半
    x1 = tl.load(X + ROTARY_DIM_HALF*stride_x_headdim, mask=mask, other=0.0).to(tl.float32)    # 后半

    # 二维旋转（每对通道）：
    # [x0', x1'] = [ x0*cos - x1*sin,  x0*sin + x1*cos ]
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos

    # 回写到 OUT 的前半/后半
    tl.store(OUT,                                   o0, mask=mask)
    tl.store(OUT + ROTARY_DIM_HALF*stride_out_headdim, o1, mask=mask)

```

这是 NeoX-style 的“前一半 vs 后一半”成对旋转实现。store 时同样支持掩码与广播规则

## 交错（GPT-J）路径：按偶奇交错配对旋转
```py
else:
    rk = tl.arange(0, BLOCK_K)
    X   = X   + (rh[:,None,None]*stride_x_nheads + rm[None,:,None]*stride_x_seqlen + rk[None,None,:]*stride_x_headdim)
    OUT = OUT + (rh[:,None,None]*stride_out_nheads+ rm[None,:,None]*stride_out_seqlen+ rk[None,None,:]*stride_out_headdim)
    mask = (rh[:,None,None] < nheads) & (rm[None,:,None] < seqlen) & (rk[None,None,:] < ROTARY_DIM)

    # 一次把 BLOCK_K 宽读进来
    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)

    # 把最后一维 reshape 成 (..., BLOCK_K//2, 2)，再按最后一维=2切成 (x0, x1)
    x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    o  = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])

    tl.store(OUT, o, mask=mask)

```
GPT-J-style（交错/奇偶配对）：把最后一维改成 [..., BLOCK_K//2, 2]，每个“2”即一对通道（偶/奇）做旋转；再用 join/reshape 复原。

其中，tl.reshape / tl.split / tl.join 是 Triton 在块张量上的形状/拼接原语。



