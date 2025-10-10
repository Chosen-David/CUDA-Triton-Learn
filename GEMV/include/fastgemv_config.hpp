#pragma once
#include <vector>
#include <cstddef>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef SHARED_MEM_MAX_ROWS
#define SHARED_MEM_MAX_ROWS 64
#endif

namespace fg {

inline unsigned ceil_div(unsigned v, unsigned d) { return (v + d - 1) / d; }

// 默认 block（仅用于打印/基线，成本模型里你已经支持 block 搜索）
struct BlockCfg { unsigned x = 32, y = 4; };

// sweep 配置（供 test 程序使用）
struct SweepCfg {
  std::vector<size_t> Ms; // 例如 2048..20480
  std::vector<size_t> Ks; // 2048..50000，8 对齐
  int warmup_iters = 4;
  int measure_iters = 10;
};

} // namespace fg
