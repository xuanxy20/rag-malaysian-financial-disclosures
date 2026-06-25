import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

metrics = ["Context\nPrecision", "Context\nRecall", "Faithfulness", "Answer\nRelevancy"]

path_a = [0.887, 0.624, 0.573, 0.271]
path_b = [0.825, 0.594, 0.594, 0.533]
path_c = [0.942, 0.667, 0.646, 0.476]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5.5))

colors = ["#4C72B0", "#DD8452", "#55A868"]  # muted blue, orange, green

bars_a = ax.bar(x - width, path_a, width, label="Path A (Baseline)",      color=colors[0], edgecolor="white", linewidth=0.8)
bars_b = ax.bar(x,         path_b, width, label="Path B (Layout-Aware)",   color=colors[1], edgecolor="white", linewidth=0.8)
bars_c = ax.bar(x + width, path_c, width, label="Path C (Hybrid)",         color=colors[2], edgecolor="white", linewidth=0.8)

# value labels on top of each bar
for bars in [bars_a, bars_b, bars_c]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=7.5, color="#333333"
        )

# mark the winner per metric group with a subtle star
winners = [2, 2, 2, 1]  # index of winning path bar group (0=A,1=B,2=C)
winner_bars = [bars_a, bars_b, bars_c]
for i, w in enumerate(winners):
    bar = winner_bars[w][i]
    ax.annotate(
        "★",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.045),
        ha="center", va="bottom",
        fontsize=9, color="#333333"
    )

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10.5)
ax.set_ylabel("RAGAS Score", fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="grey")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# legend placed below the x-axis labels, horizontal
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
          fontsize=9.5, framealpha=0.85)

# note placed top-right, well above all bars
ax.annotate(
    "★ = highest metric score",
    xy=(0.98, 0.93), xycoords="axes fraction",
    ha="right", va="top", fontsize=8, color="#555555",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8)
)

plt.title("FIN Questions (n=18): RAGAS Scores by Pipeline", fontsize=12.5, pad=12)
plt.tight_layout()

out_path = "results/reports/fin_ragas_bar.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
