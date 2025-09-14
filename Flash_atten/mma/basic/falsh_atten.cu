//ref: https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/flash-attn/mma/basic/flash_attn_mma_share_kv.cu
//解读：https://editor.csdn.net/md/?articleId=148876441

#include "utils.h"

/*
输入QKV的维度 [B, H, N, d]：
B = batch size
H = num_heads (QKV_head)
N = 序列长度 （QKV_seqlen）
d = head_dim (kHeadDim)



kMmaTileSeqLenQ其实可以理解成Q的WARP_ROWS，

*/
template <
    const int kHeadDim,          // Headdim, 32,64,128
    const int kMmaTileSeqLenQ,   // 4, more MMA(warp), M=16*4=64, Q@K^T=[Br(M),
                                 // d(K)]@[d(K),  Bc(N)]
    const int kMmaTileSeqLenK,   // 1, more MMA(warp), N=8*1 =8,  Q@K^T=[Br(M),
                                 // d(K)]@[d(K),  Bc(N)]
    const int kMmaTileSeqLenP,   // 4, more MMA(warp), M=16*4=64, P@V
                                 // =[Br(M),Bc(K)]@[Bc(K), d(N) ]
    const int kMmaTileHeadDimV,  // 1, more MMA(warp), N=8*1 =8,  P@V
                                 // =[Br(M),Bc(K)]@[Bc(K), d(N) ]
    const int kWarpTileSeqLenQ,  // 1, more values, M, Br=64*1=64, matmul M
    const int kWarpTileSeqLenK,  // 8, more values, N, Bc=8*8 =64, matmul N
    const int kWarpTileSeqLenP,  // 1, more values, M, Br=64*1=64, matmul M
    const int kWarpTileHeadDimV, // 8, more values, N,
                                 // d=8*(1|2|3|4|...)=8|...|32|64|96|128|...
    const int kOStorageAccFloat32, // 0/1, MMA Acc always be fp16, but O
                                   // storage can be fp32 or half.
    const int kStage,              // 1,2
    const int kPadQ,               // Pad Q/K/V 0,8
    const int kPadK, const int kPadV>
__global__ void __launch_bounds__(WARP_SIZE *kMmaTileSeqLenQ *kMmaTileSeqLenK)
    flash_attn_mma_stages_split_q_shared_kv_kernel(half *Q, half *K, half *V,
                                                   half *O, int QKV_seqlen,
                                                   int QKV_head) {
            



}