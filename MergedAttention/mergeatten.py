import torch
# ref:https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_merge_attn_states.py
# ref:https://zhuanlan.zhihu.com/p/1892966682634473987

# CPU版本 这里的shape适配的是vllm

# 在同一个 head 上，把 该 head 在 prefix 部分 和 该 head 在 suffix 部分 的注意力输出 + log-sum-exp 合并为一个最终的 head 输出。
def merge_attn_states_torch(
        output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        prefix_output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        prefix_lse: torch.Tensor,  # [NUM_HEADS, NUM_TOKENS]
        suffix_output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        suffix_lse: torch.Tensor,  # [NUM_HEADS, NUM_TOKENS]
        output_lse: Optional[torch.Tensor] = None,  # [NUM_HEADS, NUM_TOKENS]
):
   # 分别代表 prefix 部分 和 suffix 部分 的 log-sum-exp（LSE）值
   p_lse = prefix_lse
   s_lse = suffix_lse
   # 注意这里是主元素比较和替换，防止数值溢出
   p_lse[p_lse == torch.inf] = -torch.inf
   s_lse[s_lse == torch.inf] = -torch.inf
   max_lse = torch.maximum(p_lse, s_lse)
   p_lse -= max_lse
   s_lse -= max_lse

   # output_lse 是一个可选参数
   if output_lse is not None:
        output_lse = torch.log(torch.exp(p_lse) + torch.exp(s_lse)) + max_lse
    
   # 下面计算scale，也即softmax的部分
   p_scale = torch.exp(p_lse) / (torch.exp(p_lse) + torch.exp(s_lse))
   s_scale = torch.exp(s_lse) / (torch.exp(p_lse) + torch.exp(s_lse))

   # 将原本的形状 [NUM_HEADS, NUM_TOKENS] 对应到 输出张量的形状 [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
   p_scale = torch.transpose(p_scale, 0, 1).unsqueeze(-1)
   s_scale = torch.transpose(s_scale, 0, 1).unsqueeze(-1)

   # 对结果校准得到最终Attention输出
   output = prefix_output * p_scale + suffix_output * s_scale

   #返回结果 
   return output, output_lse

   

    
     




