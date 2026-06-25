import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────

metrics = ["Context Precision", "Context Recall", "Faithfulness", "Answer Relevancy"]

# By Layout Sensitivity
layout = {
    "HIGH (n=25)": {
        "A": [0.899, 0.643, 0.641, 0.424],
        "B": [0.829, 0.629, 0.671, 0.521],
        "C": [0.890, 0.707, 0.702, 0.545],
    },
    "MEDIUM (n=4)": {
        "A": [0.488, 0.396, 0.799, 0.090],
        "B": [0.497, 0.521, 0.700, 0.362],
        "C": [0.767, 0.792, 0.876, 0.595],
    },
    "LOW (n=1)": {
        "A": [0.804, 0.800, 0.250, 0.627],
        "B": [1.000, 0.800, 0.250, 0.679],
        "C": [1.000, 0.250, None, 0.000],
    },
}

# By Question Type
qtype = {
    "FIN (n=18)": {
        "A": [0.887, 0.624, 0.573, 0.271],
        "B": [0.825, 0.594, 0.594, 0.533],
        "C": [0.942, 0.667, 0.646, 0.476],
    },
    "RISK (n=2)": {
        "A": [0.250, 0.000, 0.000, 0.000],
        "B": [0.458, 0.125, 0.500, 0.187],
        "C": [0.574, 0.750, 0.667, 0.644],
    },
    "GOV (n=3)": {
        "A": [0.983, 0.867, 0.715, 0.505],
        "B": [0.903, 0.869, 0.750, 0.659],
        "C": [0.781, 0.774, 0.875, 0.791],
    },
    "OPS (n=4)": {
        "A": [0.951, 0.750, 0.850, 0.788],
        "B": [0.899, 0.792, 0.582, 0.329],
        "C": [0.952, 0.825, 0.846, 0.433],
    },
    "SUS (n=2)": {
        "A": [0.500, 0.458, None, 0.619],
        "B": [0.312, 0.583, 1.000, 0.770],
        "C": [0.531, 0.625, None, 0.749],
    },
    "REG (n=1)": {
        "A": [1.000, 0.714, 0.833, 0.798],
        "B": [1.000, 0.714, 1.000, 0.346],
        "C": [1.000, 0.714, 0.833, 0.553],
    },
}

# ── Colours ───────────────────────────────────────────────────────────────────

layout_colors = {
    "HIGH (n=25)":   ["#2166AC", "#4393C3", "#92C5DE"],   # dark→light blue
    "MEDIUM (n=4)":  ["#D6604D", "#F4A582", "#FDDBC7"],   # dark→light orange
    "LOW (n=1)":     ["#888888", "#AAAAAA", "#CCCCCC"],   # greys
}

qtype_colors = {
    "FIN (n=18)":  ["#1F77B4", "#4C9FD4", "#A8D1EF"],
    "RISK (n=2)":  ["#D62728", "#EF6C6D", "#F5B8B8"],
    "GOV (n=3)":   ["#2CA02C", "#5DC25D", "#AADCAA"],
    "OPS (n=4)":   ["#9467BD", "#B899D7", "#D9C2EC"],
    "SUS (n=2)":   ["#17BECF", "#52CDD8", "#A4E2E9"],
    "REG (n=1)":   ["#FF7F0E", "#FFB05A", "#FFD4A3"],
}

# ── Helper ────────────────────────────────────────────────────────────────────

def plot_panel(ax, groups_data, colors_map, metric_idx, title):
    """Draw one subplot for one metric column."""
    paths = ["A", "B", "C"]
    group_names = list(groups_data.keys())
    n_groups = len(group_names)
    n_paths  = 3
    total_bars = n_groups * n_paths
    group_width = 0.8
    bar_w = group_width / n_paths

    x_positions = np.arange(n_paths)   # one cluster per path

    for gi, gname in enumerate(group_names):
        clrs = colors_map[gname]
        for pi, path in enumerate(paths):
            val = groups_data[gname][path][metric_idx]
            xpos = pi + (gi - (n_groups - 1) / 2) * bar_w
            if val is None:
                continue
            bar = ax.bar(xpos, val, bar_w * 0.92,
                         color=clrs[pi], edgecolor="white", linewidth=0.5)
            # data label
            ax.text(xpos, val + 0.012, f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=5.2, color="#222222", rotation=90,
                    fontweight="normal")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Path A", "Path B", "Path C"], fontsize=8)
    ax.set_ylim(0, 1.22)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="grey")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle("Stratified RAGAS Results — Path A vs B vs C",
             fontsize=14, fontweight="bold", y=1.01)

# ── Row 1: By Layout Sensitivity ──────────────────────────────────────────────

for col, metric in enumerate(metrics):
    ax = axes[0, col]
    plot_panel(ax, layout, layout_colors, col, metric)
    if col == 0:
        ax.set_ylabel("Score", fontsize=9)

# Row 1 legend
layout_patches = []
for gname, clrs in layout_colors.items():
    layout_patches.append(mpatches.Patch(color=clrs[0], label=gname))
axes[0, 0].legend(handles=layout_patches, title="Layout sensitivity",
                  fontsize=7, title_fontsize=7.5, loc="upper right",
                  framealpha=0.85)

# Row 1 label
fig.text(0.01, 0.73, "By Layout\nSensitivity",
         va="center", ha="left", fontsize=9, fontweight="bold",
         rotation=90, color="#333333")

# ── Row 2: By Question Type ───────────────────────────────────────────────────

for col, metric in enumerate(metrics):
    ax = axes[1, col]
    plot_panel(ax, qtype, qtype_colors, col, metric)
    if col == 0:
        ax.set_ylabel("Score", fontsize=9)

# Row 2 legend
qtype_patches = []
for gname, clrs in qtype_colors.items():
    qtype_patches.append(mpatches.Patch(color=clrs[0], label=gname))
axes[1, 0].legend(handles=qtype_patches, title="Question type",
                  fontsize=7, title_fontsize=7.5, loc="upper right",
                  framealpha=0.85)

# Row 2 label
fig.text(0.01, 0.27, "By Question\nType",
         va="center", ha="left", fontsize=9, fontweight="bold",
         rotation=90, color="#333333")

plt.tight_layout(rect=[0.03, 0, 1, 1])

out_path = "results/reports/stratified_bar_chart_labeled.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
