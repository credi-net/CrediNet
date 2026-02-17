
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_label_distribution_by_group(
    df: pd.DataFrame,
    group_map: dict[str, list[str]],
    output_dir: Path,
    normalize: bool = True,
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
        bar_kwargs = {"edgecolor": "black", "linewidth": 0.6}
        ax.bar([0], [value_0], color="#d9534f", label="0", **bar_kwargs)
        ax.bar([1], [value_1], color="#5cb85c", label="1", **bar_kwargs)
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
        fig.tight_layout()
        fig.savefig(output_dir / f"label_distribution_{group_name}.png", dpi=300)
        plt.close(fig)


def plot_pc1_quantiles(
    df: pd.DataFrame,
    output_path: Path,
    quantiles: int = 4,
) -> None:
    if "pc1" not in df.columns:
        raise ValueError("Expected column 'pc1' in domain_pc1.csv")

    bins = np.linspace(0.0, 1.0, 9)
    bin_labels = [
        f"{int(round(bins[i] * 100))}-{int(round(bins[i + 1] * 100))}%"
        for i in range(len(bins) - 1)
    ]
    cut_bins = pd.cut(df["pc1"], bins=bins, labels=bin_labels, include_lowest=True)
    counts = cut_bins.value_counts().reindex(bin_labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#d9534f", "#e06c4a", "#f0ad4e", "#c7c56b", "#9ec87a", "#8ec07c", "#6fbf75", "#5cb85c"]
    x = np.arange(len(counts))
    ax.bar(
        x,
        counts.values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
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
    fig.savefig(output_path, dpi=300)
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
        default=4,
        help="Number of quantile bins for pc1",
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Plot counts instead of proportions for label distribution",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    labels_df = pd.read_csv(args.labels_url)
    plot_label_distribution_by_group(
        labels_df,
        DOMAINS,
        out_dir,
        normalize=not args.counts,
    )

    pc1_df = pd.read_csv(args.domain_pc1)
    plot_pc1_quantiles(
        pc1_df,
        out_dir / "domain_pc1_quantile_counts.png",
        quantiles=args.quantiles,
    )


if __name__ == "__main__":
    main()