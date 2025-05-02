import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

filename = "timings.txt"

# Read the data
df = pd.read_csv(filename, sep=r"\s+", header=0)

block_sizes = sorted(df["block_size"].unique())
markers = ["o", "s", "^", "d", "v", "*", "p", "h"]
colors = plt.cm.viridis([i / len(block_sizes) for i in range(len(block_sizes))])

# Computational Speedup vs. Matrix Size (Propagation)
plt.figure(figsize=(10, 6))
for i, bs in enumerate(block_sizes):
    subset = df[df["block_size"] == bs]
    plt.plot(
        subset["n"],
        subset["prop_speedup"],
        marker=markers[i % len(markers)],
        color=colors[i],
        linestyle="-",
        label=f"Block Size {int(bs)}",
    )
plt.xlabel(r"Matrix Size ($n$, where $n=m$)")
plt.ylabel("Speedup")
plt.title("Computational Speedup vs. Matrix Size (Propagation)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xscale("log", base=2)
plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
plt.xticks(sorted(df["n"].unique()))
plt.tight_layout()
plt.savefig("figs/prop-speedup-vs-size.png")
plt.close()

# CPU and GPU Computation Time vs. Matrix Size (Propagation)
plt.figure(figsize=(10, 6))
for i, bs in enumerate(block_sizes):
    subset = df[df["block_size"] == bs]
    if i == 0:
        plt.plot(
            subset["n"],
            subset["prop_cpu_time"],
            marker="x",
            color="black",
            linestyle="--",
            label=f"CPU Time",
        )
    plt.plot(
        subset["n"],
        subset["prop_gpu_time"],
        marker=markers[i % len(markers)],
        color=colors[i],
        linestyle="-",
        label=f"GPU Time (Block Size {int(bs)})",
    )
plt.xlabel(r"Matrix Size ($n$, where $n=m$)")
plt.ylabel("Time (seconds)")
plt.title("CPU and GPU Computation Time vs. Matrix Size (Propagation)")
plt.yscale("log")
plt.xscale("log", base=2)
plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
plt.xticks(sorted(df["n"].unique()))
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("figs/prop-times-vs-size.png")
plt.close()

# Computational Speedup vs. Matrix Size (Averaging)
plt.figure(figsize=(10, 6))
for i, bs in enumerate(block_sizes):
    subset = df[df["block_size"] == bs]
    plt.plot(
        subset["n"],
        subset["average_speedup"],
        marker=markers[i % len(markers)],
        color=colors[i],
        linestyle="-",
        label=f"Block Size {int(bs)}",
    )
plt.xlabel(r"Matrix Size ($n$, where $n=m$)")
plt.ylabel("Speedup")
plt.title("Computational Speedup vs. Matrix Size (Averaging)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xscale("log", base=2)
plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
plt.xticks(sorted(df["n"].unique()))
plt.tight_layout()
plt.savefig("figs/avg-speedup-vs-size.png")
plt.close()

# CPU and GPU Computation Time vs. Matrix Size (Averaging)
plt.figure(figsize=(10, 6))
for i, bs in enumerate(block_sizes):
    subset = df[df["block_size"] == bs]
    if i == 0:
        plt.plot(
            subset["n"],
            subset["average_cpu_time"],
            marker="x",
            color="black",
            linestyle="--",
            label=f"CPU Time",
        )
    plt.plot(
        subset["n"],
        subset["average_gpu_time"],
        marker=markers[i % len(markers)],
        color=colors[i],
        linestyle="-",
        label=f"GPU Time (Block {int(bs)})",
    )
plt.xlabel(r"Matrix Size ($n$, where $n=m$)")
plt.ylabel("Time (seconds)")
plt.title("CPU and GPU Computation Time vs. Matrix Size (Propagation)")
plt.yscale("log")
plt.xscale("log", base=2)
plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
plt.xticks(sorted(df["n"].unique()))
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("figs/avg-times-vs-size.png")
plt.close()

# Maximum Differences between CPU and GPU vs. Matrix Size
plt.figure(figsize=(10, 6))
for i, bs in enumerate(block_sizes):
    subset = df[df["block_size"] == bs]
    plt.plot(
        subset["n"],
        subset["prop_max_diff"],
        marker=markers[i % len(markers)],
        color=colors[i],
        linestyle="-",
        label=f"Prop. Diff. (Block Size {int(bs)})",
    )
    plt.plot(
        subset["n"],
        subset["average_max_diff"],
        marker=markers[i % len(markers)],
        markerfacecolor="none",  # Use hollow marker
        color=colors[i],
        linestyle="--",  # Dashed line
        label=f"Avg. Diff. (Block Size {int(bs)})",
    )
plt.xlabel(r"Matrix Size ($n$, where $n=m$)")
plt.ylabel("Difference")
plt.title("Maximum Differences between CPU and GPU vs. Matrix Size")
plt.yscale("log")
plt.xscale("log", base=2)
plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
plt.xticks(sorted(df["n"].unique()))
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("figs/maxdiff-vs-size.png")
plt.close()
