from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use Times New Roman for the research figures
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12


# ============================================================
# Verified experimental results
# ============================================================

# Figure 1: Baseline model comparison
baseline_models = [
    "Proposed\nPruned DT",
    "Random\nForest",
    "Linear\nSVM",
    "Compact\nNN",
]

baseline_f1 = [
    0.952294,
    0.956080,
    0.491910,
    0.624880,
]


# Figure 2: Ablation study
ablation_labels = [
    "A\nFull\nunpruned",
    "B\nCorrelation-\nreduced",
    "C\nFull\npruned",
    "D\nSelected +\npruned",
]

ablation_f1 = [
    0.945814,
    0.935994,
    0.956265,
    0.952294,
]


# Figure 3: Attack-category detection
attack_categories = [
    "Backdoor",
    "DoS",
    "Exploits",
    "Reconnaissance",
]

attack_detection = [
    98.97,
    97.19,
    96.47,
    97.94,
]


# Figure 4: Permutation importance
feature_names = [
    "sttl",
    "smean",
    "dload",
    "proto_tcp",
    "ct_dst_src_ltm",
    "dmean",
    "ct_srv_src",
    "ct_srv_dst",
    "proto_udp",
    "ct_state_ttl",
    "dpkts",
    "sload",
    "dinpkt",
    "dur",
    "sjit",
]

permutation_importance = [
    0.339480,
    0.138317,
    0.098705,
    0.086977,
    0.086372,
    0.086238,
    0.075278,
    0.072550,
    0.056863,
    0.055143,
    0.054756,
    0.039204,
    0.036738,
    0.031302,
    0.012828,
]


# ============================================================
# Figure 1 — Baseline Model Comparison
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5.5))

bars = ax.bar(
    baseline_models,
    baseline_f1,
)

ax.set_title(
    "Baseline Model Comparison",
    fontsize=14,
)

ax.set_ylabel(
    "F1-score",
    fontsize=12,
)

ax.set_ylim(0, 1.05)

ax.grid(
    axis="y",
    alpha=0.3,
)

for bar, value in zip(bars, baseline_f1):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure_1_baseline_comparison.pdf",
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# Figure 2 — Ablation Study
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5.5))

bars = ax.bar(
    ablation_labels,
    ablation_f1,
)

ax.set_title(
    "Ablation Study: F1-score",
    fontsize=14,
)

ax.set_ylabel(
    "F1-score",
    fontsize=12,
)

# Updated axis: full F1-score range
ax.set_ylim(0.0, 1.0)

ax.grid(
    axis="y",
    alpha=0.3,
)

for bar, value in zip(bars, ablation_f1):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure_2_ablation_f1.pdf",
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# Figure 3 — Attack-Category Detection Performance
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5.5))

bars = ax.bar(
    attack_categories,
    attack_detection,
)

ax.set_title(
    "Attack-Category Detection Performance",
    fontsize=14,
)

# Updated wording
ax.set_ylabel(
    "Attack detection rate (%)",
    fontsize=12,
)

# Updated axis: complete percentage range
ax.set_ylim(0, 100)

ax.grid(
    axis="y",
    alpha=0.3,
)

for bar, value in zip(bars, attack_detection):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.0,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11,
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure_3_attack_category_detection.pdf",
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# Figure 4 — Permutation Importance
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

# Reverse order so the largest feature appears at the top
features_plot = feature_names[::-1]
importance_plot = permutation_importance[::-1]

bars = ax.barh(
    features_plot,
    importance_plot,
)

ax.set_title(
    "Top 15 Permutation-Importance Features",
    fontsize=14,
)

ax.set_xlabel(
    "Mean decrease in F1-score after permutation",
    fontsize=12,
)

ax.grid(
    axis="x",
    alpha=0.3,
)

for bar, value in zip(bars, importance_plot):
    ax.text(
        value + 0.004,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        fontsize=10,
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure_4_permutation_importance.pdf",
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# Completion message
# ============================================================

print()
print("All figures generated successfully.")
print(f"Saved to: {OUTPUT_DIR}")