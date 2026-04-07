from functools import lru_cache
import csv
import json
import re
from pathlib import Path
from threading import Lock
from urllib.parse import quote
from urllib.parse import urlparse

import duckdb
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download

try:
    import tldextract
except Exception:  # pragma: no cover - optional dependency fallback
    tldextract = None

DATASET_REPO = "credi-net/CrediPred"
DOMAINREL_DATASET_REPO = "credi-net/DomainRel"
REGRESSION_FILENAME = "mlpInfer_dec2024_pc1_embeddinggemma-300m_GNN-RNI.parquet"
BINARY_FILENAME = "MLP-Inference/Text+GAT/mlpInfer_binaryClassifcation_dec2024_embeddinggemma-300m_GAT-text_scores.parquet"
BINARY_LABELS_FILENAME = "labels.csv"
REGRESSION_SOURCE = f"https://huggingface.co/datasets/{DATASET_REPO}/blob/main/{quote(REGRESSION_FILENAME, safe='/')}"
BINARY_SOURCE = f"https://huggingface.co/datasets/{DATASET_REPO}/blob/main/{quote(BINARY_FILENAME, safe='/')}"
DOMAINREL_SOURCE = f"https://huggingface.co/datasets/{DOMAINREL_DATASET_REPO}/blob/main/{BINARY_LABELS_FILENAME}"
DQR_LABELS_PATH = Path(__file__).parent / "data" / "domain_pc1.csv"
DQR_SOURCE = "data/domain_pc1.csv"

API_VERSION = "0.4.1"
DATA_CUTOFF_MONTH = "2024-12"
METHOD = "Content + topology-based"

MONTHS_MANIFEST_PATH = Path(__file__).with_name("months.json")

app = FastAPI(title="CrediGraph API", version=API_VERSION)

con = None
reg_path = bin_path = None
domain_col = "domain"
reg_score_col = "pc1_score"
bin_score_col = "binary_score"

label_sets_catalog: dict = {"label_sets": []}
binary_label_lookup: dict[str, bool] = {}
regression_label_lookup: dict[str, float] = {}
DOMAIN_REGEX = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,}$")
_db_lock = Lock()

if tldextract is not None:
    _extract = tldextract.TLDExtract(include_psl_private_domains=True)


def _canonicalize_domain(value: str) -> str:
    if not value or not isinstance(value, str):
        raise ValueError("Domain must be a non-empty string")

    value = value.strip().lower()
    if not value.startswith(("http://", "https://")):
        value = "http://" + value

    parsed = urlparse(value)
    host = parsed.hostname or ""
    if host.startswith("www."):
        host = host[4:]

    if not DOMAIN_REGEX.match(host):
        raise ValueError(f"Invalid domain: {host}")

    return host


def _flip_regression_domain_key(domain: str) -> str:
    """Convert canonical host to regression parquet key format."""
    domain = domain.strip(".").lower()
    if not domain:
        return domain

    if tldextract is not None:
        ext = _extract(domain)
        if ext.domain and ext.suffix:
            return f"{ext.suffix}.{ext.domain}"

    # Fallback when tldextract is unavailable.
    parts = domain.split(".")
    if len(parts) < 2:
        return domain
    return f"{parts[-1]}.{'.'.join(parts[:-1])}"

def _parse_binary_label(value: str | None) -> bool | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized == "":
        return None
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Unsupported binary label value: {value}")


def _load_binary_label_lookup(path: str | Path) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                domain = _canonicalize_domain(row["domain"])
            except ValueError:
                # DomainRel can include non-domain values (e.g., IPs); ignore those rows.
                continue
            label = _parse_binary_label(row.get("bin"))
            if label is not None:
                labels[domain] = label
    return labels


def _load_regression_label_lookup(path: str | Path) -> dict[str, float]:
    labels: dict[str, float] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("pc1")
            if value is None or value == "":
                continue
            try:
                domain = _canonicalize_domain(row["domain"])
            except ValueError:
                continue
            labels[domain] = float(value)
    return labels


def _build_label_sets_catalog() -> dict:
    return {
        "label_sets": [
            {
                "name": "DomainRel",
                "task": "binary",
                "label_field": "bin",
                "description": "Binary ground-truth credibility labels for GT internal queries.",
                "source_url": DOMAINREL_SOURCE,
                "format": "csv",
                "domain_count": len(binary_label_lookup),
            },
            {
                "name": "DQR",
                "task": "regression",
                "label_field": "pc1",
                "description": "Continuous ground-truth credibility labels for GT internal queries.",
                "source_url": DQR_SOURCE,
                "format": "csv",
                "domain_count": len(regression_label_lookup),
            },
        ]
    }


def _build_prediction_result(
    domain: str,
    reg: float | None,
    binary: bool | int | None,
) -> dict | None:
    resolved_reg = reg
    resolved_binary = None if binary is None else bool(binary)

    if resolved_reg is None and resolved_binary is None:
        return None

    out = {"domain": domain}
    if resolved_reg is not None:
        out["credibility_level"] = round(float(resolved_reg), 2)
    if resolved_binary is not None:
        out["credible"] = bool(resolved_binary)
    return out


def _build_ground_truth_result(domain: str) -> dict | None:
    reg = regression_label_lookup.get(domain)
    binary = binary_label_lookup.get(domain)

    if reg is None and binary is None:
        return None

    out = {"domain": domain}
    if reg is not None:
        out["credibility_level"] = round(float(reg), 2)
    if binary is not None:
        out["credible"] = bool(binary)
    return out


@app.on_event("startup")
def startup():
    global con, reg_path, bin_path
    global label_sets_catalog
    global binary_label_lookup, regression_label_lookup

    reg_path = hf_hub_download(repo_id=DATASET_REPO, filename=REGRESSION_FILENAME, repo_type="dataset")
    bin_path = hf_hub_download(repo_id=DATASET_REPO, filename=BINARY_FILENAME, repo_type="dataset")
    binary_labels_path = hf_hub_download(
        repo_id=DOMAINREL_DATASET_REPO,
        filename=BINARY_LABELS_FILENAME,
        repo_type="dataset",
    )

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=2")

    binary_label_lookup = _load_binary_label_lookup(binary_labels_path)
    regression_label_lookup = _load_regression_label_lookup(DQR_LABELS_PATH)
    label_sets_catalog = _build_label_sets_catalog()


@lru_cache(maxsize=200_000)
def _lookup(domain: str):
    with _db_lock:
        reg_val = bin_val = None

        row = con.execute(
            f"SELECT {reg_score_col} FROM read_parquet(?) WHERE {domain_col} = ? LIMIT 1",
            [reg_path, domain],
        ).fetchone()
        if not row:
            flipped = _flip_regression_domain_key(domain)
            if flipped != domain:
                row = con.execute(
                    f"SELECT {reg_score_col} FROM read_parquet(?) WHERE {domain_col} = ? LIMIT 1",
                    [reg_path, flipped],
                ).fetchone()
        if row:
            reg_val = row[0]

        row = con.execute(
            f"SELECT {bin_score_col} FROM read_parquet(?) WHERE {domain_col} = ? LIMIT 1",
            [bin_path, domain],
        ).fetchone()
        if row:
            bin_val = row[0]

        return domain, reg_val, bin_val


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


@app.get("/label_sets", include_in_schema=False)
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
                "domain_count": item.get("domain_count"),
            }
            for item in label_sets_catalog["label_sets"]
        ]
    }


@app.post("/by_domain")
def by_domain(
    domain: str,
):
    # Single domain, prediction binary-only mode.
    try:
        cleaned = _canonicalize_domain(domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        matched, _, binary = _lookup(cleaned)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    if binary is None:
        raise HTTPException(status_code=404, detail="Domain not found")

    return {
        "domain": matched or cleaned,
        "credible": bool(binary),
    }


@app.post("/by_domains")
def by_domains(
    domains: list[str],
):
    # Batch domains, prediction binary-only mode.
    results = []
    for domain in domains:
        try:
            cleaned = _canonicalize_domain(domain)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            matched, _, binary = _lookup(cleaned)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Internal server error") from exc

        out = {"domain": matched or cleaned}
        if binary is not None:
            out["credible"] = bool(binary)
        results.append(out)
    return results


@app.post("/by_domain_cts")
def _by_domain_cts(domain: str):
    # Single domain, prediction continuous-only mode.
    try:
        cleaned = _canonicalize_domain(domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        matched, reg, _ = _lookup(cleaned)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    if reg is None:
        raise HTTPException(status_code=404, detail="Domain not found")

    return {
        "domain": matched or cleaned,
        "credibility_level": round(float(reg), 2),
    }


@app.post("/by_domains_cts")
def _by_domains_cts(domains: list[str]):
    # Batch domains, prediction continuous-only mode.
    results = []
    for domain in domains:
        try:
            cleaned = _canonicalize_domain(domain)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            matched, reg, _ = _lookup(cleaned)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Internal server error") from exc

        out = {"domain": matched or cleaned}
        if reg is not None:
            out["credibility_level"] = round(float(reg), 2)
        results.append(out)
    return results


@app.post("/by_domain_dr")
def _by_domain_dr(domain: str):
    # Single domain, DomainRel binary-only mode.
    try:
        cleaned = _canonicalize_domain(domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    binary = binary_label_lookup.get(cleaned)
    if binary is None:
        raise HTTPException(status_code=404, detail="Domain not found")

    return {
        "domain": cleaned,
        "credible": bool(binary),
    }


@app.post("/by_domains_dr")
def _by_domains_dr(domains: list[str]):
    # Batch domains, DomainRel binary-only mode.
    results = []
    for domain in domains:
        try:
            cleaned = _canonicalize_domain(domain)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        out = {"domain": cleaned}
        binary = binary_label_lookup.get(cleaned)
        if binary is not None:
            out["credible"] = bool(binary)
        results.append(out)
    return results


@app.post("/by_domain_dqr")
def _by_domain_dqr(domain: str):
    # Single domain, GT regression-only mode (DQR pc1).
    try:
        cleaned = _canonicalize_domain(domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    reg = regression_label_lookup.get(cleaned)
    if reg is None:
        raise HTTPException(status_code=404, detail="Domain not found")

    return {
        "domain": cleaned,
        "credibility_level": round(float(reg), 2),
    }


@app.post("/by_domains_dqr")
def _by_domains_dqr(domains: list[str]):
    # Batch domains, GT regression-only mode (DQR pc1).
    results = []
    for domain in domains:
        try:
            cleaned = _canonicalize_domain(domain)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        out = {"domain": cleaned}
        reg = regression_label_lookup.get(cleaned)
        if reg is not None:
            out["credibility_level"] = round(float(reg), 2)
        results.append(out)
    return results



