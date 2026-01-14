import re
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    classification_report,
)

from pathlib import Path


KNOWN_SUFFIXES = {
    "com", "org", "net", "edu", "gov", "io",
    "co.uk", "org.uk", "ac.uk",
    "co.jp", "com.au", "org.au",
}

SCORES_PATH = "scores/inferred/mlpInfer_dec2024_pc1_embeddinggemma-300m_GNN-RNI.parquet"
LABELS_PATH = "scores/labels/labels.csv"
LABEL_DIR = "scores/labels/classification/processed"
OUTPUT_PREFIX = "eval"
THRESHOLD = 0.5

SCORE_COL = "pc1_score"
REG_COL = "reg_score"
WEAK_COL = "weak_label"

DISPLAY_NAMES = {
    SCORE_COL: "predicted score",
    REG_COL: "score (DQR)",
}

def flip_if_needed(domain: str) -> str:
    if pd.isna(domain):
        return domain
    domain = domain.strip().lower()
    if "." not in domain:
        return domain
    parts = domain.split(".")
    for s in KNOWN_SUFFIXES:
        if domain.endswith(s):
            return domain
    return ".".join(reversed(parts))


def normalize(domain: str) -> str:
    if pd.isna(domain):
        return domain
    domain = domain.strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    return domain.split("/")[0]

def load_labels():
    dfs = []
    for path in Path(LABEL_DIR).glob("*.csv"):
        df = pd.read_csv(path)
        df["source"] = path.stem
        df["norm_domain"] = df["domain"].apply(normalize)
        df = df.rename(columns={"label": WEAK_COL})
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def calibration_curve(cls_df, bins=10):
    df = cls_df.copy()
    df["bin"] = pd.cut(df[SCORE_COL], bins=bins, labels=False)

    grouped = df.groupby("bin")
    calib = grouped.agg(
        mean_score=(SCORE_COL, "mean"),
        empirical_rate=(WEAK_COL, "mean"),
        count=(WEAK_COL, "size"),
    ).dropna()

    print("\nCalibration table:")
    print(calib)

    plt.figure(figsize=(6, 6))
    plt.plot(calib["mean_score"], calib["empirical_rate"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel(f"Mean {DISPLAY_NAMES[SCORE_COL]}")
    plt.ylabel("Empirical P(reliable)")
    plt.title("Calibration curve")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_calibration_curve.png", dpi=150)
    plt.close()

    return calib

def expected_calibration_error(calib):
    w = calib["count"] / calib["count"].sum()
    ece = (w * (calib["mean_score"] - calib["empirical_rate"]).abs()).sum()
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    return ece

def inspect_errors(cls_df, k=20):
    fp = cls_df[(cls_df[WEAK_COL] == 1) & (cls_df["pred"] == 0)]
    fn = cls_df[(cls_df[WEAK_COL] == 0) & (cls_df["pred"] == 1)]

    print(f"\nFalse positives (pred=unreliable, true=reliable): {len(fp)}")
    print(fp[["norm_domain", "source", SCORE_COL]].sort_values(SCORE_COL).head(k))

    print(f"\nFalse negatives (pred=reliable, true=unreliable): {len(fn)}")
    print(fn[["norm_domain", "source", SCORE_COL]].sort_values(SCORE_COL, ascending=False).head(k))

def plot_pr_curve(cls_df):
    p, r, _ = precision_recall_curve(cls_df[WEAK_COL], cls_df[SCORE_COL])

    plt.figure(figsize=(6, 6))
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_pr_curve.png", dpi=150)
    plt.close()

def evaluate_by_source(cls_df):
    print("\nPer-dataset breakdown:\n")

    rows = []
    for source, g in cls_df.groupby("source"):
        if len(g) < 50:
            continue

        y_true = g[WEAK_COL]
        y_pred = g["pred"]
        y_score = g[SCORE_COL]

        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score) if len(y_true.unique()) > 1 else float("nan")
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)

        rows.append({
            "source": source,
            "n": len(g),
            "acc": acc,
            "auc": auc,
            "prec_0": p[0], "rec_0": r[0], "f1_0": f[0],
            "prec_1": p[1], "rec_1": r[1], "f1_1": f[1],
        })

    report = pd.DataFrame(rows).sort_values("n", ascending=False)
    print(report.to_string(index=False, float_format="%.3f"))

def stream_scores(label_set):
    pf = pq.ParquetFile(SCORES_PATH)
    chunks = []
    seen = matched = 0
    for i in range(pf.num_row_groups):
        chunk = pf.read_row_group(i).to_pandas()
        seen += len(chunk)
        chunk["norm_domain"] = chunk["domain"].apply(flip_if_needed).apply(normalize)
        hit = chunk[chunk["norm_domain"].isin(label_set)]
        matched += len(hit)
        if not hit.empty:
            chunks.append(hit)
        if i % 50 == 0:
            print(f"Scanned {seen:,} rows — matched {matched}")
    return pd.concat(chunks, ignore_index=True)


def print_label_distribution(labels_df):
    dist = labels_df[WEAK_COL].value_counts(dropna=False).sort_index()
    frac = dist / dist.sum()
    print("\nWeak label distribution (before merge):")
    print(pd.DataFrame({"count": dist, "fraction": frac}))


def evaluate(df):
    reg_df = df.dropna(subset=[REG_COL])
    cls_df = df.dropna(subset=[WEAK_COL]).copy()
    cls_df["pred"] = (cls_df[SCORE_COL] >= THRESHOLD).astype(int)

    print(f"\nEvaluation set size: {len(df)}")

    if not reg_df.empty:
        print("\nRegression")
        print(f"MSE: {mean_squared_error(reg_df[REG_COL], reg_df[SCORE_COL]):.4f}")
        print(f"MAE: {mean_absolute_error(reg_df[REG_COL], reg_df[SCORE_COL]):.4f}")
        print(f"R²:  {r2_score(reg_df[REG_COL], reg_df[SCORE_COL]):.4f}")
    else:
        print("\nNo regression data available")

    if not cls_df.empty:
        acc = accuracy_score(cls_df[WEAK_COL], cls_df["pred"])
        auc = roc_auc_score(cls_df[WEAK_COL], cls_df[SCORE_COL])
        p, r, f, _ = precision_recall_fscore_support(cls_df[WEAK_COL], cls_df["pred"], zero_division=0)

        print("\nClassification")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC AUC:  {auc:.4f}")

        print("\nPer-class precision / recall:")
        print(pd.DataFrame({
            "precision": p,
            "recall": r,
            "f1": f
        }, index=["unreliable (0)", "reliable (1)"]))

        print("\nFull classification report:")
        print(classification_report(cls_df[WEAK_COL], cls_df["pred"], digits=3))
    else:
        print("\nNo classification data available")

    return reg_df, cls_df


def make_plots(reg_df, cls_df):
    sns.set(style="whitegrid")

    if not reg_df.empty:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=reg_df, x=REG_COL, y=SCORE_COL, alpha=0.4)
        plt.xlabel(DISPLAY_NAMES[REG_COL])
        plt.ylabel(DISPLAY_NAMES[SCORE_COL])
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_PREFIX}_regression_scatter.png", dpi=150)
        plt.close()

    if not cls_df.empty:
        cls_df["abs_error"] = (cls_df[SCORE_COL] - cls_df[WEAK_COL]).abs()

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=cls_df, x=WEAK_COL, y="abs_error")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_PREFIX}_error_by_label.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.histplot(data=cls_df, x=SCORE_COL, hue=WEAK_COL, bins=50, kde=True, element="step")
        plt.xlabel(DISPLAY_NAMES[SCORE_COL])
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_PREFIX}_score_by_label.png", dpi=150)
        plt.close()


def main():
    labels_df = load_labels()
    print_label_distribution(labels_df)

    label_set = set(labels_df["norm_domain"].dropna())
    scores_df = stream_scores(label_set)

    df = scores_df.merge(labels_df, on="norm_domain", how="inner")
    print(f"\nJoined rows: {len(df)}")

    reg_df, cls_df = evaluate(df)
    evaluate_by_source(cls_df)
    make_plots(reg_df, cls_df)

    labels_df["in_graph"] = labels_df["norm_domain"].isin(label_set)
    print("\nGraph coverage by source:")
    print(labels_df.groupby("source")["in_graph"].mean())

    calib = calibration_curve(cls_df)
    expected_calibration_error(calib)

    inspect_errors(cls_df)
    plot_pr_curve(cls_df)

if __name__ == "__main__":
    main()