#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

#include "../gemv_smem_cost_model.cuh"

namespace smemlog {

struct CacheEntry {
  unsigned int tile_k_half = 0;
  unsigned int k_stage = 1;
  double best_gflops = 0.0;
  size_t shared_bytes = 0;
};

using CacheMap = std::unordered_map<std::string, CacheEntry>;

// 生成 cache key: "M x K"
std::string make_key(size_t rows, size_t K);

// 读写 cache
CacheMap& get_cache();                 // 懒加载 json / legacy
void persist_cache(const CacheMap&);   // 写 gemv_smem_cost_cache.json

// 追加搜索日志（把 block 维度也写入）
void append_trials(size_t rows, size_t K, const std::vector<GemvSmemTrialTrace>& trials);

// 对外接口：优先返回缓存；未命中时调用 gemv_smem_cost_model(...) 搜索，写日志/缓存
CacheEntry query_or_search(size_t rows, size_t K, cudaStream_t stream);

} // namespace smemlog
