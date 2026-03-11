"""
Add or update one month in months.json with precomputed graph and score stats.

Computations are done once offline and stored in months.json:
- nodes / edges
- overlap with previous month (shared nodes / shared edges)
- overlap with DomainRel domains
- regression score mean and quartiles
- disagreement count between binary and regression scores

Usage:
    python scripts/compute_graph_stats.py --month-url https://huggingface.co/datasets/credi-net/CrediBench/tree/main/dec2024 --month 2024-12 --update months.json

    python scripts/compute_graph_stats.py --month-url https://huggingface.co/datasets/credi-net/CrediBench/tree/main/jan2025 --month 2025-01 --previous-month 2024-12 --update months.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import requests

CREDIBENCH_REPO = "credi-net/CrediBench"
CREDIPRED_REPO = "credi-net/CrediPred"
DOMAINREL_REPO = "credi-net/DomainRel"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
HF_API_TREE = "https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
HF_HEADERS = {"User-Agent": "credinet-month-ingestion/1.0"}

DOMAINREL_PARQUETS = [
    "regression/train_regression_domains.parquet",
    "regression/val_regression_domains.parquet",
    "regression/test_regression_domains.parquet",
]


@dataclass
class ScoreFiles:
    regression_url: str
    binary_url: str


def _http_get_json(url: str) -> list[dict]:
    resp = requests.get(url, headers=HF_HEADERS, timeout=120)
    resp.raise_for_status()
    return resp.json()


def month_to_folder(month: str) -> str:
    return datetime.strptime(month, "%Y-%m").strftime("%b%Y").lower()


def folder_to_month(folder: str) -> str:
    dt = datetime.strptime(folder, "%b%Y")
    return dt.strftime("%Y-%m")


def parse_month_folder_from_url(month_url: str) -> str:
    p = urlparse(month_url)
    path = p.path.rstrip("/")
    if "/tree/main/" in path:
        return path.split("/tree/main/")[-1].split("/")[0]
    return path.split("/")[-1]


def hf_resolve(repo: str, rel_path: str) -> str:
    return HF_RESOLVE.format(repo=repo, path=rel_path)


def _decompress_stream(resp: requests.Response):
    d = zlib.decompressobj(zlib.MAX_WBITS | 16)
    leftover = b""
    for chunk in resp.iter_content(chunk_size=1 << 20):
        block = leftover + d.decompress(chunk)
        lines = block.split(b"\n")
        leftover = lines[-1]
        yield from lines[:-1]
    if leftover:
        yield leftover


def count_lines(url: str) -> int:
    resp = requests.get(url, headers=HF_HEADERS, stream=True, timeout=600)
    resp.raise_for_status()
    count, first = 0, True
    for line in _decompress_stream(resp):
        if first:
            first = False
            continue
        if line:
            count += 1
    return count


def stream_column(url: str, col_index: int, out_path: str) -> None:
    resp = requests.get(url, headers=HF_HEADERS, stream=True, timeout=600)
    resp.raise_for_status()
    first = True
    with open(out_path, "wb") as fout:
        for line in _decompress_stream(resp):
            if first:
                first = False
                continue
            if not line:
                continue
            parts = line.split(b",")
            if col_index < len(parts):
                fout.write(parts[col_index].strip() + b"\n")


def stream_edge_keys(url: str, out_path: str) -> None:
    resp = requests.get(url, headers=HF_HEADERS, stream=True, timeout=600)
    resp.raise_for_status()
    first = True
    with open(out_path, "wb") as fout:
        for line in _decompress_stream(resp):
            if first:
                first = False
                continue
            if not line:
                continue
            parts = line.split(b",")
            if len(parts) >= 2:
                fout.write(parts[0].strip() + b"," + parts[1].strip() + b"\n")


def header_col_index(url: str, preferred_names: list[str]) -> int:
    resp = requests.get(url, headers=HF_HEADERS, stream=True, timeout=120)
    resp.raise_for_status()
    for line in _decompress_stream(resp):
        if not line:
            continue
        cols = [c.strip().decode() for c in line.split(b",")]
        for c in preferred_names:
            if c in cols:
                return cols.index(c)
        return 0
    raise RuntimeError(f"Could not read CSV header from {url}")


def sort_unique(src: str, dst: str) -> None:
    subprocess.run(["sort", "-u", "-o", dst, src], check=True)


def count_common_lines(path_a: str, path_b: str) -> int:
    proc = subprocess.run(["comm", "-12", path_a, path_b], capture_output=True, check=True)
    return proc.stdout.count(b"\n")


def find_score_files(month_folder: str) -> ScoreFiles:
    entries = _http_get_json(HF_API_TREE.format(repo=CREDIPRED_REPO))
    paths = [e.get("path", "") for e in entries]

    reg_candidates = [
        p
        for p in paths
        if "MLP-Inference/Text+GAT/" in p and f"reg_{month_folder}" in p and p.endswith(".parquet")
    ]
    bin_candidates = [
        p
        for p in paths
        if "MLP-Inference/Text+GAT/" in p and f"binaryClassifcation_{month_folder}" in p and p.endswith(".parquet")
    ]

    if not reg_candidates:
        reg_candidates = [
            p for p in paths if "MLP-Inference/Text+GAT/" in p and f"reg_{month_folder}" in p and ".parquet" in p
        ]
    if not bin_candidates:
        bin_candidates = [
            p
            for p in paths
            if "MLP-Inference/Text+GAT/" in p and f"binaryClassifcation_{month_folder}" in p and ".parquet" in p
        ]

    if not reg_candidates or not bin_candidates:
        raise RuntimeError(
            f"Could not infer score parquet files for {month_folder}. "
            "Provide --regression-url and --binary-url explicitly."
        )

    return ScoreFiles(
        regression_url=hf_resolve(CREDIPRED_REPO, sorted(reg_candidates)[0]),
        binary_url=hf_resolve(CREDIPRED_REPO, sorted(bin_candidates)[0]),
    )


def score_stats(con: duckdb.DuckDBPyConnection, score_files: ScoreFiles) -> dict:
    con.execute("LOAD httpfs")

    reg_cols = [
        r[0]
        for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [score_files.regression_url]).fetchall()
    ]
    bin_cols = [
        r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [score_files.binary_url]).fetchall()
    ]

    reg_domain = next((c for c in ["domain", "Domain", "host"] if c in reg_cols), None)
    bin_domain = next((c for c in ["domain", "Domain", "host"] if c in bin_cols), None)
    reg_score = next((c for c in ["continuous_score", "pc1_score", "pc1", "score"] if c in reg_cols), None)
    bin_score = next((c for c in ["binary_score", "binary_classification_score", "prediction", "score"] if c in bin_cols), None)

    if not reg_domain or not bin_domain or not reg_score or not bin_score:
        raise RuntimeError(
            f"Unable to detect required columns. reg={reg_cols}, bin={bin_cols}"
        )

    q = f"""
    WITH reg AS (
        SELECT {reg_domain} AS domain, CAST({reg_score} AS DOUBLE) AS reg_score
        FROM read_parquet(?)
    ),
    bin AS (
        SELECT {bin_domain} AS domain, CAST({bin_score} AS DOUBLE) AS bin_score
        FROM read_parquet(?)
    ),
    reg_stats AS (
        SELECT
            AVG(reg_score) AS reg_mean,
            quantile_cont(reg_score, 0.25) AS q1,
            quantile_cont(reg_score, 0.50) AS q2,
            quantile_cont(reg_score, 0.75) AS q3
        FROM reg
    ),
    joined AS (
        SELECT r.reg_score, b.bin_score
        FROM reg r
        JOIN bin b USING (domain)
    )
    SELECT
        rs.reg_mean,
        rs.q1,
        rs.q2,
        rs.q3,
        SUM(
            CASE
                WHEN (CASE WHEN bin_score >= 0.5 THEN 1 ELSE 0 END) = 1 AND reg_score < rs.reg_mean THEN 1
                WHEN (CASE WHEN bin_score >= 0.5 THEN 1 ELSE 0 END) = 0 AND reg_score > rs.reg_mean THEN 1
                ELSE 0
            END
        ) AS disagreement_count
    FROM joined, reg_stats rs
    """

    row = con.execute(q, [score_files.regression_url, score_files.binary_url]).fetchone()
    reg_mean, q1, q2, q3, disagreement_count = row

    return {
        "regression_score_stats": {
            "mean": float(reg_mean),
            "quartiles": {
                "q1": float(q1),
                "q2": float(q2),
                "q3": float(q3),
            },
        },
        "regression_binary_disagreement_count": int(disagreement_count or 0),
    }


def compute_graph_counts(month_folder: str) -> tuple[int, int]:
    v_url = hf_resolve(CREDIBENCH_REPO, f"{month_folder}/vertices.csv.gz")
    e_url = hf_resolve(CREDIBENCH_REPO, f"{month_folder}/edges.csv.gz")
    nodes = count_lines(v_url)
    edges = count_lines(e_url)
    return nodes, edges


def build_sorted_graph_keys(month_folder: str, tmpdir: str) -> tuple[str, str]:
    v_url = hf_resolve(CREDIBENCH_REPO, f"{month_folder}/vertices.csv.gz")
    e_url = hf_resolve(CREDIBENCH_REPO, f"{month_folder}/edges.csv.gz")

    node_idx = header_col_index(v_url, ["domain", "Domain", "host", "node_id", "nid"])

    v_raw = os.path.join(tmpdir, f"v_raw_{month_folder}.txt")
    e_raw = os.path.join(tmpdir, f"e_raw_{month_folder}.txt")
    v_sorted = os.path.join(tmpdir, f"v_sorted_{month_folder}.txt")
    e_sorted = os.path.join(tmpdir, f"e_sorted_{month_folder}.txt")

    stream_column(v_url, node_idx, v_raw)
    stream_edge_keys(e_url, e_raw)
    sort_unique(v_raw, v_sorted)
    sort_unique(e_raw, e_sorted)
    return v_sorted, e_sorted


def previous_month_overlap(curr_folder: str, prev_folder: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        curr_v, curr_e = build_sorted_graph_keys(curr_folder, tmpdir)
        prev_v, prev_e = build_sorted_graph_keys(prev_folder, tmpdir)
        return {
            "previous_month": folder_to_month(prev_folder),
            "shared_nodes": count_common_lines(curr_v, prev_v),
            "shared_edges": count_common_lines(curr_e, prev_e),
        }


def build_sorted_domainrel_nodes(con: duckdb.DuckDBPyConnection, out_sorted: str) -> int:
    con.execute("LOAD httpfs")
    urls = [hf_resolve(DOMAINREL_REPO, p) for p in DOMAINREL_PARQUETS]

    raw_path = out_sorted + ".raw"
    con.execute(
        """
        COPY (
            SELECT DISTINCT domain
            FROM read_parquet(?)
            WHERE domain IS NOT NULL
        ) TO ? (HEADER false)
        """,
        [urls, raw_path],
    )
    sort_unique(raw_path, out_sorted)
    os.remove(raw_path)

    proc = subprocess.run(["wc", "-l", out_sorted], capture_output=True, check=True, text=True)
    return int(proc.stdout.strip().split()[0])


def domainrel_overlap(con: duckdb.DuckDBPyConnection, curr_folder: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        curr_v, _ = build_sorted_graph_keys(curr_folder, tmpdir)
        domainrel_sorted = os.path.join(tmpdir, "domainrel_sorted.txt")
        domainrel_nodes = build_sorted_domainrel_nodes(con, domainrel_sorted)
        shared = count_common_lines(curr_v, domainrel_sorted)

    return {
        "shared_nodes": shared,
        "domainrel_nodes": domainrel_nodes,
    }


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"months": []}
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        payload = {"months": payload}
    if not isinstance(payload.get("months"), list):
        raise RuntimeError("Manifest must contain top-level 'months' list")
    return payload


def infer_previous_month_folder(month: str) -> str:
    dt = datetime.strptime(month, "%Y-%m")
    y = dt.year
    m = dt.month - 1
    if m == 0:
        y -= 1
        m = 12
    prev = datetime(y, m, 1)
    return prev.strftime("%b%Y").lower()


def add_month_to_manifest(
    month_url: str,
    month: str,
    manifest_path: Path,
    previous_month: str | None = None,
    regression_url: str | None = None,
    binary_url: str | None = None,
    label: str | None = None,
    description: str | None = None,
    download_url: str | None = None,
    data_format: str = "parquet",
) -> dict:
    month_folder = parse_month_folder_from_url(month_url)

    if month != folder_to_month(month_folder):
        raise ValueError(f"month={month} does not match month folder {month_folder}")

    scores = ScoreFiles(regression_url=regression_url, binary_url=binary_url)
    if not regression_url or not binary_url:
        scores = find_score_files(month_folder)

    con = duckdb.connect()

    nodes, edges = compute_graph_counts(month_folder)

    if previous_month is None:
        prev_folder = infer_previous_month_folder(month)
        try:
            ov_prev = previous_month_overlap(month_folder, prev_folder)
        except Exception:
            ov_prev = {"previous_month": folder_to_month(prev_folder), "shared_nodes": None, "shared_edges": None}
    else:
        prev_folder = month_to_folder(previous_month)
        ov_prev = previous_month_overlap(month_folder, prev_folder)

    ov_domainrel = domainrel_overlap(con, month_folder)
    score_payload = score_stats(con, scores)

    item = {
        "month": month,
        "label": label or datetime.strptime(month, "%Y-%m").strftime("%B %Y scores"),
        "download_url": download_url or month_url,
        "size_bytes": None,
        "format": data_format,
        "description": description or f"Scores and graph stats for {month}.",
        "nodes": nodes,
        "edges": edges,
        "overlap_with_previous_month": ov_prev,
        "overlap_with_domainrel": ov_domainrel,
        "score_sources": {
            "regression": scores.regression_url,
            "binary": scores.binary_url,
        },
        **score_payload,
    }

    manifest = load_manifest(manifest_path)
    months = manifest["months"]
    for i, m in enumerate(months):
        if m.get("month") == month:
            months[i] = {**m, **item}
            break
    else:
        months.append(item)

    months.sort(key=lambda x: x.get("month", ""))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month-url", required=True, help="CrediBench month URL, e.g. .../tree/main/dec2024")
    parser.add_argument("--month", required=True, help="Month in YYYY-MM format")
    parser.add_argument("--update", default="months.json", help="Path to months.json")
    parser.add_argument("--previous-month", default=None, help="Previous month in YYYY-MM format")
    parser.add_argument("--regression-url", default=None, help="Override regression parquet URL")
    parser.add_argument("--binary-url", default=None, help="Override binary parquet URL")
    parser.add_argument("--label", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--download-url", default=None)
    args = parser.parse_args()

    item = add_month_to_manifest(
        month_url=args.month_url,
        month=args.month,
        manifest_path=Path(args.update),
        previous_month=args.previous_month,
        regression_url=args.regression_url,
        binary_url=args.binary_url,
        label=args.label,
        description=args.description,
        download_url=args.download_url,
    )

    print(json.dumps(item, indent=2))


if __name__ == "__main__":
    main()
