#include <fstream>
#include <iostream>
#include "bench_writer.hpp"

namespace bench {

bool write_csv(const std::string& path, const std::vector<Row>& rows) {
  std::ofstream f(path);
  if (!f) { std::cerr<<"[csv] open "<<path<<" failed\n"; return false; }
  f << "M,K,kernel,avg_ms,gflops\n";
  for (const auto& r: rows)
    f << r.M << ',' << r.K << ',' << r.kernel << ','
      << r.avg_ms << ',' << r.gflops << '\n';
  return true;
}

} // namespace bench
