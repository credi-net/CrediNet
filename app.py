from functools import lru_cache
import os
import secrets
import json
from pathlib import Path
import re
from urllib.parse import quote, urlparse

import duckdb
from fastapi import Depends, FastAPI, Header, HTTPException
from huggingface_hub import hf_hub_download
import tldextract

DATASET_REPO = "credi-net/CrediPred"
REGRESSION_FILENAME = "MLP-Inference/Text+GAT/mlpInfer_reg_dec2024_embeddinggemma-300m_GAT-text_scores.parquet.parquet"
BINARY_FILENAME = "MLP-Inference/Text+GAT/mlpInfer_binaryClassifcation_dec2024_embeddinggemma-300m_GAT-text_scores.parquet"
REGRESSION_SOURCE = f"https://huggingface.co/datasets/{DATASET_REPO}/blob/main/{quote(REGRESSION_FILENAME, safe='/')}"
BINARY_SOURCE = f"https://huggingface.co/datasets/{DATASET_REPO}/blob/main/{quote(BINARY_FILENAME, safe='/')}"

API_VERSION = "0.2.2"
DATA_CUTOFF_MONTH = "2024-12"
METHOD = "GAT-TEXT"

MONTHS_MANIFEST_PATH = Path(__file__).with_name("months.json")
LABEL_SETS_MANIFEST_PATH = Path(__file__).with_name("label_sets.json")

app = FastAPI(title="CrediNet API", version=API_VERSION)

con = None
reg_path = bin_path = None
reg_domain_col = reg_score_col = bin_domain_col = bin_score_col = None

months_catalog: dict = {"months": []}
label_sets_catalog: dict = {"label_sets": []}

_extract = tldextract.TLDExtract(include_psl_private_domains=True)
_DOMAIN_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,}$")


def _require_internal_access(
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> None:
    expected = os.getenv("CREDINET_INTERNAL_TOKEN")
    if not expected:
        # Fail closed when token is not configured.
        raise HTTPException(status_code=503, detail="Internal endpoints are disabled")
    if not x_internal_token or not secrets.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=403, detail="Forbidden")


def _canonicalize(value: str) -> str:
    if not value or not isinstance(value, str):
        raise ValueError("Domain must be a non-empty string")
    value = value.strip().lower()
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    host = urlparse(value).hostname or ""
    if host.startswith("www."):
        host = host[4:]
    if not _DOMAIN_RE.match(host):
        raise ValueError(f"Invalid domain: {host}")
    return host


def _flip(domain: str) -> str:
    domain = domain.strip(".").lower()
    ext = _extract(domain)
    return f"{ext.suffix}.{ext.domain}" if ext.domain and ext.suffix else domain


def _unflip(domain: str) -> str:
    labels = domain.strip(".").lower().split(".")
    for split in range(len(labels) - 1, 0, -1):
        suffix = ".".join(labels[:split])
        candidate = ".".join(labels[split:] + labels[:split])
        if _extract(candidate).suffix == suffix:
            return candidate
    return domain


def _variants(value: str) -> list[str]:
    normalized = _canonicalize(value)
    canonical = _unflip(normalized)
    flipped = _flip(canonical)
    seen = set()
    result = []
    for candidate in (normalized, canonical, flipped):
        if candidate and candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def _discover_columns(path: str, score_candidates: list[str]) -> tuple[str, str]:
    cols = [r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [path]).fetchall()]
    domain_col = next((c for c in ["domain", "Domain", "host"] if c in cols), None)
    score_col = next((c for c in score_candidates if c in cols), None)
    if not domain_col:
        raise RuntimeError(f"No domain column found. Got: {cols}")
    if not score_col:
        raise RuntimeError(f"No score column found. Got: {cols}")
    return domain_col, score_col


def _load_months_manifest() -> dict:
    if not MONTHS_MANIFEST_PATH.exists():
        return {"months": []}
    with MONTHS_MANIFEST_PATH.open() as f:
        payload = json.load(f)
    if isinstance(payload, list):
        payload = {"months": payload}
    if not isinstance(payload.get("months"), list):
        raise RuntimeError("months.json must contain a top-level 'months' list")
    return payload


def _load_label_sets_manifest() -> dict:
    if not LABEL_SETS_MANIFEST_PATH.exists():
        return {"label_sets": []}
    with LABEL_SETS_MANIFEST_PATH.open() as f:
        payload = json.load(f)
    if isinstance(payload, list):
        payload = {"label_sets": payload}
    if not isinstance(payload.get("label_sets"), list):
        raise RuntimeError("label_sets.json must contain a top-level 'label_sets' list")
    return payload


@app.on_event("startup")
def startup():
    global con, reg_path, bin_path
    global reg_domain_col, reg_score_col, bin_domain_col, bin_score_col
    global months_catalog, label_sets_catalog

    reg_path = hf_hub_download(repo_id=DATASET_REPO, filename=REGRESSION_FILENAME, repo_type="dataset")
    bin_path = hf_hub_download(repo_id=DATASET_REPO, filename=BINARY_FILENAME, repo_type="dataset")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=2")

    reg_domain_col, reg_score_col = _discover_columns(
        reg_path, ["continuous_score", "pc1_score", "pc1", "score"]
    )
    bin_domain_col, bin_score_col = _discover_columns(
        bin_path, ["binary_score", "binary_classification_score", "prediction", "score"]
    )

    months_catalog = _load_months_manifest()
    label_sets_catalog = _load_label_sets_manifest()


@lru_cache(maxsize=200_000)
def _lookup(domain: str):
    matched = reg_val = bin_val = None
    for candidate in _variants(domain):
        if reg_val is None:
            row = con.execute(
                f"SELECT {reg_score_col} FROM read_parquet(?) WHERE {reg_domain_col} = ? LIMIT 1",
                [reg_path, candidate],
            ).fetchone()
            if row:
                reg_val = row[0]
                matched = matched or candidate

        if bin_val is None:
            row = con.execute(
                f"SELECT {bin_score_col} FROM read_parquet(?) WHERE {bin_domain_col} = ? LIMIT 1",
                [bin_path, candidate],
            ).fetchone()
            if row:
                bin_val = row[0]
                matched = matched or candidate

        if reg_val is not None and bin_val is not None:
            break

    return matched, reg_val, bin_val


@app.get("/")
def root():
    return {"status": "ok", "api_version": API_VERSION, "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "api_version": API_VERSION}


@app.get("/metadata")
def metadata():
    return {
        "api_version": API_VERSION,
        "data_cutoff_month": DATA_CUTOFF_MONTH,
        "method": METHOD,
        "score_sources": {
            "regression": REGRESSION_SOURCE,
            "binary": BINARY_SOURCE,
        },
    }


@app.get("/label_sets")
def label_sets():
    return {
        "label_sets": [
            {
                "name": item["name"],
                "task": item["task"],
                "label_field": item["label_field"],
                "description": item.get("description"),
                "source_url": item.get("source_url"),
                "format": item.get("format"),
                "splits": item.get("splits", []),
                "domain_count": item.get("domain_count"),
            }
            for item in label_sets_catalog["label_sets"]
        ]
    }


@app.get("/by_domain/{domain}")
def by_domain(domain: str):
    try:
        matched, reg, binary = _lookup(domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if reg is None and binary is None:
        raise HTTPException(status_code=404, detail="Domain not found")

    out = {"domain": matched or domain}
    if reg is not None:
        out["continuous_score"] = round(float(reg), 2)
        out["pc1_score"] = float(reg)
    if binary is not None:
        out["binary_score"] = round(float(binary), 2)
    return out


@app.get("/stats", dependencies=[Depends(_require_internal_access)], include_in_schema=False)
def stats():
    return {
        "months": [
            {
                "month": m["month"],
                "nodes": m["nodes"],
                "edges": m["edges"],
                "overlap_with_previous_month": m["overlap_with_previous_month"],
                "overlap_with_domainrel": m.get("overlap_with_domainrel"),
                "regression_score_stats": m.get("regression_score_stats"),
                "regression_binary_disagreement_count": m.get("regression_binary_disagreement_count"),
            }
            for m in months_catalog["months"]
        ]
    }


@app.get("/months", dependencies=[Depends(_require_internal_access)], include_in_schema=False)
def months():
    return {
        "months": [
            {
                "month": m["month"],
                "label": m["label"],
                "download_url": m["download_url"],
                "size_bytes": m.get("size_bytes"),
                "format": m.get("format"),
                "description": m.get("description"),
            }
            for m in months_catalog["months"]
        ]
    }
