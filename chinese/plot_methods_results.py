import matplotlib.pyplot as plt


def main():
    methods = ["Gloss-RoBERTa", "OHPT", "SenseRAG"]
    accuracies = [0.6360544217687075, 0.37244897959183676, 0.679]
    spearman = [0.331847343281402, 0.24909859533892303, 0.472]

    metrics = ["Accuracy", "Spearman"]
    metric_positions = [0, 1.1]
    bar_width = 0.22
    colors = ["#6baed6", "#fdae6b", "#9e9ac8"]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    acc_bars = []
    sp_bars = []
    offsets = [(-1 + i) * bar_width for i in range(len(methods))]

    for i, (label, acc, sp) in enumerate(zip(methods, accuracies, spearman)):
        acc_bar = ax1.bar(
            metric_positions[0] + offsets[i],
            acc,
            width=bar_width,
            color=colors[i],
            label=label,
        )
        acc_bars.append(acc_bar[0])

        sp_bar = ax2.bar(
            metric_positions[1] + offsets[i],
            sp,
            width=bar_width,
            color=colors[i],
        )
        sp_bars.append(sp_bar[0])

    ax1.set_xticks(metric_positions)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax2.set_ylabel("Spearman Correlation", fontsize=12)
    ax1.set_title("Gloss-RoBERTa, OHPT, SenseRAG Results", fontsize=14)
    ax1.grid(axis="y", alpha=0.3)

    ax1.set_ylim(0, 0.75)
    ax2.set_ylim(0, 0.55)

    ax1.legend(acc_bars, methods, loc="upper right", ncol=3, frameon=True)

    for bar, val in zip(acc_bars, accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for bar, val in zip(sp_bars, spearman):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    plt.savefig("methods_results_bar.png", dpi=150)
    print("Saved methods_results_bar.png")


if __name__ == "__main__":
    main()
