"""Plot the CSV written by `make bench`.

usage: python scripts/plot.py results/bench.csv [out.png]

Draws GFLOPS per kernel across sizes, plus % of cuBLAS at the largest size,
and prints a Markdown results table ready to paste into the README.
"""

import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def shape_label(m, n, k):
    return m if m == n == k else f"{m}x{n}x{k}"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".png"

    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{src} has no results")

    shapes = []
    by_kernel = defaultdict(dict)  # (id, name) -> shape -> row
    for r in rows:
        shape = (r["M"], r["N"], r["K"])
        if shape not in shapes:
            shapes.append(shape)
        by_kernel[(int(r["kernel_id"]), r["kernel"])][shape] = r
    largest = max(shapes, key=lambda s: int(s[0]) * int(s[1]) * int(s[2]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    for (kid, name), results in sorted(by_kernel.items()):
        xs = [i for i, s in enumerate(shapes) if s in results]
        ys = [float(results[shapes[i]]["gflops"]) for i in xs]
        ax1.plot(xs, ys, marker="o", label=f"{kid} {name}")
    ax1.set_xticks(range(len(shapes)), [shape_label(*s) for s in shapes])
    ax1.set_xlabel("matrix size")
    ax1.set_ylabel("GFLOPS (log scale)")
    ax1.set_yscale("log")
    ax1.set_title("SGEMM throughput")
    ax1.grid(alpha=0.3)
    ax1.legend()

    bars = [
        (f"{kid} {name}", float(results[largest]["pct_cublas"]))
        for (kid, name), results in sorted(by_kernel.items())
        if largest in results
    ]
    ax2.barh([b[0] for b in bars], [b[1] for b in bars])
    ax2.invert_yaxis()
    ax2.set_xlabel("% of cuBLAS")
    ax2.set_title(f"At {shape_label(*largest)}")
    for i, (_, pct) in enumerate(bars):
        ax2.text(pct, i, f" {pct:.1f}%", va="center")
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}\n")

    print(f"| # | kernel | GFLOPS @ {shape_label(*largest)} | % of cuBLAS |")
    print("|---|---|---|---|")
    for (kid, name), results in sorted(by_kernel.items()):
        if largest in results:
            r = results[largest]
            print(f"| {kid} | {name} | {float(r['gflops']):.0f} | {float(r['pct_cublas']):.1f}% |")


if __name__ == "__main__":
    main()
