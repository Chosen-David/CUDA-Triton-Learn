#include "smem_log.hpp"
#include <fstream>
#include <regex>
#include <sstream>
#include <iomanip>
#include <iostream>

// 引入成本模型和辅助工具
#include "../gemv_smem_cost_model.cuh"

namespace {
constexpr const char* kCacheFile  = "gemv_smem_cost_cache.json";
constexpr const char* kLegacyFile = "gemv_smem_cost_cache.txt";
constexpr const char* kSearchLog  = "gemv_smem_cost_search.json";

// 解析 cache key
bool parse_key(const std::string& key, size_t& M, size_t& K) {
  auto p = key.find('x'); if (p==std::string::npos) return false;
  try { M = std::stoull(key.substr(0,p)); K = std::stoull(key.substr(p+1)); }
  catch (...) { return false; }
  return true;
}
} // namespace

namespace smemlog {

std::string make_key(size_t rows, size_t K) {
  return std::to_string(rows) + "x" + std::to_string(K);
}

// ---- cache 懒加载 ----
static CacheMap& cache_singleton() {
  static CacheMap cache; static bool inited=false;
  if (inited) return cache;
  inited = true;

  // 尝试从 json 读取
  if (std::ifstream fin{kCacheFile}; fin.is_open()) {
    std::ostringstream buf; buf << fin.rdbuf();
    const std::string s = buf.str();
    const std::regex re(
      R"(\{\s*"M"\s*:\s*(\d+)\s*,\s*"K"\s*:\s*(\d+)\s*,\s*"tile_k_half"\s*:\s*(\d+)\s*,\s*"k_stage"\s*:\s*(\d+)\s*,\s*"best_gflops"\s*:\s*([-+]?([0-9]*[.])?[0-9]+([eE][-+]?[0-9]+)?)\s*(?:,\s*"shared_bytes"\s*:\s*(\d+))?\s*\})"
    );
    for (std::sregex_iterator it(s.begin(), s.end(), re), ed; it!=ed; ++it) {
      size_t M=0,K=0; CacheEntry e{};
      try {
        M = std::stoull((*it)[1].str());
        K = std::stoull((*it)[2].str());
        e.tile_k_half  = std::stoul((*it)[3].str());
        e.k_stage      = std::stoul((*it)[4].str());
        e.best_gflops  = std::stod((*it)[5].str());
        if ((*it)[8].matched) e.shared_bytes = std::stoull((*it)[8].str());
      } catch (...) { continue; }
      if (e.k_stage==0) e.k_stage=1;
      if (e.shared_bytes==0) e.shared_bytes = size_t(e.tile_k_half)*e.k_stage*sizeof(half);
      cache[make_key(M,K)] = e;
    }
    if (!cache.empty()) return cache;
  }

  // 兼容 legacy 文本
  if (std::ifstream fin{kLegacyFile}; fin.is_open()) {
    size_t K=0; CacheEntry e{};
    while (fin >> K >> e.tile_k_half >> e.k_stage >> e.best_gflops) {
      if (e.k_stage==0) e.k_stage=1;
      e.shared_bytes = size_t(e.tile_k_half)*e.k_stage*sizeof(half);
      cache[make_key(0,K)] = e;
    }
  }
  return cache;
}

CacheMap& get_cache() { return cache_singleton(); }

void persist_cache(const CacheMap& c) {
  std::ofstream fout(kCacheFile, std::ios::trunc);
  if (!fout) { std::cerr<<"[smem cache] open "<<kCacheFile<<" failed\n"; return; }
  fout.setf(std::ios::fixed);
  fout << "[\n";
  bool first = true;
  for (const auto& kv : c) {
    size_t M=0,K=0; if (!parse_key(kv.first, M, K)) continue;
    if (!first) fout << ",\n"; first=false;
    fout << "  {\"M\": " << M
         << ", \"K\": " << K
         << ", \"tile_k_half\": " << kv.second.tile_k_half
         << ", \"k_stage\": " << kv.second.k_stage
         << ", \"best_gflops\": " << std::setprecision(6) << kv.second.best_gflops
         << ", \"shared_bytes\": " << (unsigned long long)kv.second.shared_bytes
         << "}";
  }
  if (!first) fout << "\n";
  fout << "]\n";
}

void append_trials(size_t rows, size_t K, const std::vector<GemvSmemTrialTrace>& trials) {
  if (trials.empty()) return;
  std::ofstream fout(kSearchLog, std::ios::app);
  if (!fout) { std::cerr<<"[smem log] open "<<kSearchLog<<" failed\n"; return; }
  fout << std::fixed << std::setprecision(6);
  for (const auto& t : trials) {
    size_t smem_bytes = t.shared_bytes ? t.shared_bytes
                         : size_t(t.tile_k_half) * t.k_stage * sizeof(half);
    fout << "{\"M\": " << rows
         << ", \"K\": " << K
         << ", \"tile_k_half\": " << t.tile_k_half
         << ", \"k_stage\": " << t.k_stage
         << ", \"gflops\": " << t.gflops
         << ", \"time_ms\": " << t.time_ms
         << ", \"shared_bytes\": " << (unsigned long long)smem_bytes
         << ", \"block_dim_x\": " << t.block_x
         << ", \"block_dim_y\": " << t.block_y
         << "}\n";
  }
}

CacheEntry query_or_search(size_t rows, size_t K, cudaStream_t stream) {
  auto& cache = get_cache();
  const std::string key_exact = make_key(rows, K);
  const std::string key_fallback = make_key(0, K);

  auto it = cache.find(key_exact);
  if (it == cache.end()) it = cache.find(key_fallback);
  if (it != cache.end()) {
    auto e = it->second;
    if (e.shared_bytes==0) e.shared_bytes = size_t(e.tile_k_half)*e.k_stage*sizeof(half);
    return e;
  }

  // 未命中：调用你已有的成本模型（随机数据）
  GemvSmemCostModelResult model = gemv_smem_cost_model((unsigned)rows, (unsigned)K, stream);

  CacheEntry e{};
  e.tile_k_half  = model.tile_k_half ? model.tile_k_half : 2048u;
  e.k_stage      = model.k_stage ? model.k_stage : 2u;
  e.best_gflops  = model.best_gflops;
  e.shared_bytes = size_t(e.tile_k_half)*e.k_stage*sizeof(half);

  append_trials(rows, K, model.trials);
  cache[key_exact] = e;
  persist_cache(cache);

  std::cout << "[smem cost model] tuned " << rows << "x" << K
            << " => tile_k_half=" << e.tile_k_half
            << ", k_stage=" << e.k_stage
            << ", smem=" << e.shared_bytes
            << ", best_gflops=" << e.best_gflops
            << "\n";
  return e;
}

} // namespace smemlog
