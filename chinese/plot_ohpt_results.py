import math
import matplotlib.pyplot as plt


def main():
    systems = [
        "Majority baseline",
        "Random baseline",
        "Raw Synsets",
        "Manually-Expanded Synsets",
    ]

    accuracies = [
        0.5697278911564626,
        0.44727891156462585,
        0.336734693877551,
        0.37244897959183676,
    ]

    spearman = [
        float("nan"),
        -0.08018581545573154,
        0.20443197282626016,
        0.24909859533892303,
    ]

    metrics = ["Accuracy", "Spearman"]
    metric_positions = [0, 1.1]
    bar_width = 0.18
    colors = ["#969696", "#9ecae1", "#fdae6b", "#6a51a3"]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    acc_bars = []
    sp_bars = []
    offsets = [(-1.5 + i) * bar_width for i in range(len(systems))]

    for i, (label, acc, sp) in enumerate(zip(systems, accuracies, spearman)):
        # Accuracy cluster (left)
        acc_bar = ax1.bar(
            metric_positions[0] + offsets[i],
            acc,
            width=bar_width,
            color=colors[i],
            label=label,
        )
        acc_bars.append(acc_bar[0])

        # Spearman cluster (right)
        sp_height = 0 if math.isnan(sp) else sp
        sp_bar = ax2.bar(
            metric_positions[1] + offsets[i],
            sp_height,
            width=bar_width,
            color=colors[i],
        )
        sp_bars.append(sp_bar[0])

    ax1.set_xticks(metric_positions)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax2.set_ylabel("Spearman Correlation", fontsize=12)
    ax1.set_title("OHPT Results", fontsize=14)
    ax1.grid(axis="y", alpha=0.3)

    ax1.set_ylim(0, 0.7)
    ax2.set_ylim(-0.12, 0.3)

    ax1.legend(acc_bars, systems, loc="upper center", ncol=2, frameon=True)

    for bar, val in zip(acc_bars, accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar, val in zip(sp_bars, spearman):
        if math.isnan(val):
            continue
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    plt.savefig("ohpt_results_bar.png", dpi=150)
    print("Saved ohpt_results_bar.png")


if __name__ == "__main__":
    main()
