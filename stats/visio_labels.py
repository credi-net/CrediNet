
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde


DOMAINS = {
    "general": [
        "wikipedia",
    ],
    "misinformation": [
        "misinfo-domains",
        "nelez",
    ],
    "phishing": [
        "phish-and-legit",
        "url-phish",
        "phish-dataset",
        "legit-phish",
    ],
    "malware": [
        "urlhaus",
    ],
}


def lighten_color(hex_color: str, amount: float = 0.45) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_label_distribution_by_group(
    df: pd.DataFrame,
    group_map: dict[str, list[str]],
    output_dir: Path,
    normalize: bool = True,
    annotate_percent: bool = False,
    filename_suffix: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for group_name, columns in group_map.items():
        existing = [c for c in columns if c in df.columns]
        if not existing:
            continue
        values = df[existing].to_numpy().ravel()
        values = values[~np.isnan(values)]
        if values.size == 0:
            counts = {0.0: 0, 1.0: 0}
        else:
            unique, counts_arr = np.unique(values, return_counts=True)
            counts = {float(k): int(v) for k, v in zip(unique, counts_arr)}
        total = sum(counts.values())
        value_0 = counts.get(0.0, 0)
        value_1 = counts.get(1.0, 0)
        if normalize:
            value_0 = value_0 / total if total else 0.0
            value_1 = value_1 / total if total else 0.0

        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        bar_kwargs = {}
        base_colors = ["#d9534f", "#5cb85c"]
        if group_name != "all":
            colors = [lighten_color(c, 0.45) for c in base_colors]
        else:
            colors = base_colors
        bars = ax.bar([0, 1], [value_0, value_1], color=colors, **bar_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_ylim(0, 1 if normalize else None)
        ax.margins(x=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color("black")
        ax.tick_params(axis="both", length=0)
        if annotate_percent:
            max_val = max(value_0, value_1, 1e-9)
            y_offset = (0.02 if normalize else 0.02 * max_val)
            pct_0 = (counts.get(0.0, 0) / total * 100) if total else 0.0
            pct_1 = (counts.get(1.0, 0) / total * 100) if total else 0.0
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for bar, pct in zip(bars, [pct_0, pct_1]):
                text = ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=36,
                    fontfamily="DejaVu Sans",
                    color="black",
                )
                bar_width_px = bar.get_window_extent(renderer).width
                text_width_px = text.get_window_extent(renderer).width
                if text_width_px > 0:
                    text.set_fontsize(36)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"label_distribution_{group_name}{filename_suffix}.pdf",
            dpi=300,
            transparent=True,
        )
        plt.close(fig)


def plot_pc1_quantiles(
    df: pd.DataFrame,
    output_path: Path,
    bins_count: int = 25,
) -> None:
    if "pc1" not in df.columns:
        raise ValueError("Expected column 'pc1' in domain_pc1.csv")

    bins = np.linspace(0.0, 1.0, bins_count + 1)
    bin_labels = [
        f"{int(round(bins[i] * 100))}-{int(round(bins[i + 1] * 100))}%"
        for i in range(len(bins) - 1)
    ]
    cut_bins = pd.cut(df["pc1"], bins=bins, labels=bin_labels, include_lowest=True)
    counts = cut_bins.value_counts().reindex(bin_labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#d9534f", "#e06c4a", "#f0ad4e", "#c7c56b", "#9ec87a", "#8ec07c", "#6fbf75", "#5cb85c"]
    if len(colors) < len(counts):
        colors = (colors * (len(counts) // len(colors) + 1))[: len(counts)]
    x = np.arange(len(counts))
    ax.bar(
        x,
        counts.values,
        color=colors,
        width=1.0,
        align="edge",
    )
    ax.set_xlim(0, len(counts))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="both", length=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, transparent=True)
    plt.close(fig)


def plot_pc1_kde(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    if "pc1" not in df.columns:
        raise ValueError("Expected column 'pc1' in domain_pc1.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = df["pc1"].dropna().to_numpy()
    if values.size == 0:
        return

    x = np.linspace(0.0, 1.0, 512)
    kde = gaussian_kde(values)
    y = kde(x)

    cmap = plt.get_cmap("RdYlGn")
    for i in range(len(x) - 1):
        color = cmap(i / (len(x) - 2))
        ax.fill_between(x[i : i + 2], 0, y[i : i + 2], color=color, linewidth=0)
    ax.plot(x, y, color="#2e2e2e", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="both", length=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, transparent=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate label and pc1 plots.")
    parser.add_argument(
        "--labels-url",
        default="https://huggingface.co/datasets/credi-net/DomainRel/resolve/main/labels_annot.csv",
        help="URL for labels_annot.csv",
    )
    parser.add_argument(
        "--domain-pc1",
        default="/home/mila/k/kondrupe/CrediNet/data/regression/processed/domain_pc1.csv",
        help="Path to domain_pc1.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/mila/k/kondrupe/CrediNet/stats/figs",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=25,
        help="Number of bins for pc1",
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Plot counts instead of proportions for label distribution",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    labels_df = pd.read_csv(args.labels_url)
    numeric_cols = labels_df.select_dtypes(include=[np.number]).columns.tolist()
    all_group_cols = sorted({col for cols in DOMAINS.values() for col in cols} | set(numeric_cols))
    groups = {"all": all_group_cols, **DOMAINS}
    plot_label_distribution_by_group(
        labels_df,
        groups,
        out_dir,
        normalize=not args.counts,
    )
    plot_label_distribution_by_group(
        labels_df,
        groups,
        out_dir,
        normalize=not args.counts,
        annotate_percent=True,
        filename_suffix="_pct",
    )

    pc1_df = pd.read_csv(args.domain_pc1)
    plot_pc1_quantiles(
        pc1_df,
        out_dir / "domain_pc1_quantile_counts.pdf",
        bins_count=args.quantiles,
    )
    plot_pc1_kde(
        pc1_df,
        out_dir / "domain_pc1_kde.pdf",
    )


if __name__ == "__main__":
    main()