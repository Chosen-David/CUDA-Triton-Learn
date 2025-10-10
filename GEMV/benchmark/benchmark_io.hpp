// GEMV/benchmark/benchmark_io.hpp
#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "benchmark_types.hpp"

namespace benchio {

inline bool write_csv(const std::string& path,
                      const std::vector<BenchmarkResult>& results) {
  std::ofstream csv(path);
  if (!csv) {
    std::cerr << "Failed to open " << path << " for writing.\n";
    return false;
  }
  csv << "M,K,kernel,avg_ms,gflops\n";
  for (const auto& e : results) {
    csv << e.M << ',' << e.K << ',' << e.kernel_name << ','
        << e.avg_ms << ',' << e.gflops << '\n';
  }
  std::cout << "Wrote benchmark data to " << path << "\n";
  return true;
}

inline bool emit_plot_script(const std::string& script_path,
                             const std::string& csv_path = "gemv_performance.csv") {
  std::ofstream script(script_path);
  if (!script) {
    std::cerr << "Failed to open " << script_path << " for writing.\n";
    return false;
  }
  script
    << "import csv\n"
    << "from collections import defaultdict\n"
    << "import matplotlib\n"
    << "matplotlib.use('Agg')\n"
    << "import matplotlib.pyplot as plt\n"
    << "import sys\n"
    << "csv_path = '" << csv_path << "'\n"
    << "wanted_M = [2048 * i for i in range(1, 11)]\n"
    << "data = defaultdict(lambda: defaultdict(lambda: {'K': [], 'gflops': []}))\n"
    << "try:\n"
    << "    with open(csv_path, newline='') as f:\n"
    << "        reader = csv.DictReader(f)\n"
    << "        for row in reader:\n"
    << "            try:\n"
    << "                m = int(row['M']); k = int(row['K']); g = float(row['gflops'])\n"
    << "                kernel = row['kernel']\n"
    << "            except Exception:\n"
    << "                continue\n"
    << "            payload = data[m][kernel]\n"
    << "            payload['K'].append(k)\n"
    << "            payload['gflops'].append(g)\n"
    << "except FileNotFoundError:\n"
    << "    print(f'[plot] missing {csv_path}', file=sys.stderr)\n"
    << "    sys.exit(1)\n"
    << "for m in wanted_M:\n"
    << "    if m not in data or len(data[m]) == 0:\n"
    << "        print(f'[plot] skip M={m}: no data')\n"
    << "        continue\n"
    << "    plt.figure()\n"
    << "    kernels = data[m]\n"
    << "    drew_any = False\n"
    << "    for kernel, series in sorted(kernels.items()):\n"
    << "        if not series['K']:\n"
    << "            continue\n"
    << "        pairs = sorted(zip(series['K'], series['gflops']))\n"
    << "        ks, perf = zip(*pairs)\n"
    << "        plt.plot(ks, perf, marker='o', label=kernel)\n"
    << "        drew_any = True\n"
    << "    if not drew_any:\n"
    << "        print(f'[plot] skip M={m}: empty series for all kernels')\n"
    << "        plt.close()\n"
    << "        continue\n"
    << "    plt.xlabel('K dimension')\n"
    << "    plt.ylabel('GFLOP/s')\n"
    << "    plt.title(f'GEMV Kernel Performance (M={m})')\n"
    << "    plt.grid(True, linestyle='--', linewidth=0.5)\n"
    << "    plt.legend()\n"
    << "    plt.tight_layout()\n"
    << "    outpath = f'gemv_performance_M_{m}.png'\n"
    << "    plt.savefig(outpath, dpi=200)\n"
    << "    plt.close()\n"
    << "    print(f'[plot] wrote {outpath}')\n";
  std::cout << "Generated plotting script at " << script_path << "\n";
  return true;
}

inline void run_plot_script(const std::string& script_path) {
  int ret = std::system(("python3 " + script_path).c_str());
  if (ret != 0) {
    std::cerr << "Python plotting script exited with status " << ret << std::endl;
  } else {
    std::cout << "Generated per-M charts.\n";
  }
}

} // namespace benchio
