import math
import matplotlib.pyplot as plt


def main():
    labels = [
        "Majority baseline",
        "Random baseline",
        "Gloss-RoBERTa baseline (untrained)",
        "Token-CLS",
        "Sent-CLS",
        "Sent-CLS-WS",
    ]
    accuracies = [
        0.5697278911564626,
        0.44727891156462585,
        0.5272108843537415,
        0.5272108843537415,
        0.6207482993197279,
        0.6360544217687075,
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color=[
        "#6baed6",
        "#9ecae1",
        "#c6dbef",
        "#74c476",
        "#31a354",
        "#006d2c",
    ])
    plt.ylim(0, 0.7)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Gloss-RoBERTa Accuracy Results", fontsize=14)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{acc:.3f}",
                 ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig("gloss_roberta_results_bar.png", dpi=150)
    print("Saved gloss_roberta_results_bar.png")

    # Spearman Correlation bar chart
    spearman_raw = [
        float("nan"),
        -0.08018581545573154,
        float("nan"),
        float("nan"),
        0.3238826543017499,
        0.331847343281402,
    ]
    spearman_scores = [0.0 if math.isnan(v) else v for v in spearman_raw]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, spearman_scores, color=[
        "#fdae6b",
        "#fd8d3c",
        "#fdd0a2",
        "#9e9ac8",
        "#756bb1",
        "#54278f",
    ])
    plt.ylim(-0.15, 0.4)
    plt.ylabel("Spearman Correlation", fontsize=12)
    plt.title("Gloss-RoBERTa Spearman Results", fontsize=14)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, raw_score, plot_score in zip(bars, spearman_raw, spearman_scores):
        if math.isnan(raw_score) or plot_score == 0.0:
            continue  # skip annotating NaNs or placeholder zeros
        plt.text(bar.get_x() + bar.get_width() / 2, plot_score + 0.01, f"{plot_score:.3f}",
                 ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig("gloss_roberta_spearman_bar.png", dpi=150)
    print("Saved gloss_roberta_spearman_bar.png")


if __name__ == "__main__":
    main()
