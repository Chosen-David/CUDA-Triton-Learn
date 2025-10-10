import csv
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
csv_path = 'gemv_performance.csv'
wanted_M = [2048 * i for i in range(1, 11)]  # 2048..20480
data = defaultdict(lambda: defaultdict(lambda: {'K': [], 'gflops': []}))
try:
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                m = int(row['M']); k = int(row['K']); g = float(row['gflops'])
                kernel = row['kernel']
            except Exception:
                continue
            payload = data[m][kernel]
            payload['K'].append(k)
            payload['gflops'].append(g)
except FileNotFoundError:
    print(f'[plot] missing {csv_path}', file=sys.stderr)
    sys.exit(1)

for m in wanted_M:
    if m not in data or len(data[m]) == 0:
        print(f'[plot] skip M={m}: no data')
        continue
    plt.figure()
    kernels = data[m]
    drew_any = False
    for kernel, series in sorted(kernels.items()):
        if not series['K']:
            continue
        pairs = sorted(zip(series['K'], series['gflops']))
        ks, perf = zip(*pairs)
        plt.plot(ks, perf, marker='o', label=kernel)
        drew_any = True
    if not drew_any:
        print(f'[plot] skip M={m}: empty series for all kernels')
        plt.close()
        continue
    plt.xlabel('K dimension')
    plt.ylabel('GFLOP/s')
    plt.title(f'GEMV Kernel Performance (M={m})')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    outpath = f'gemv_performance_M_{m}.png'
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f'[plot] wrote {outpath}')
