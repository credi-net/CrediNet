# CrediGraph API Documentation

Query domain credibility using pre-trained models. Supports binary credibility classification and continuous credibility scores.

**Data Cutoff:** December 2024  
**Method:** Content + topology-based  
**API Version:** 0.4.1

---

## Installation

```bash
pip install credigraph
```

---

## Quick Start

```python
from credigraph import query, query_batch

# Single domain
result = query("apnews.com")
print(result)
# Output: {"domain": "apnews.com", "credible": True}

# Multiple domains
results = query_batch(["apnews.com", "example.com"])
for r in results:
    print(r["domain"], r["credible"])
```

---

## Core API

### `query`

```python
def query(domain: str, timeout: int = 10) -> dict
```

Query a single domain and return binary credibility prediction.

**Args:**
- `domain` (str): Domain name (e.g., "apnews.com"). Accepts URLs with http/https prefix. Automatically removes "www." prefix.
- `timeout` (int): Request timeout in seconds. Default: 10.

**Returns:**
- `dict`: `{"domain": str, "credible": bool}`

**Raises:**
- `ValueError`: If domain format is invalid.
- `requests.exceptions.Timeout`: If request exceeds timeout.
- `requests.RequestException`: For HTTP or transport errors.

**Example:**
```python
from credigraph import query

result = query("apnews.com")
if result["credible"]:
    print(f"{result['domain']} is credible")
```

---

### `query_batch`

```python
def query_batch(domains: list[str], order: str = "original", timeout: int = 10) -> list[dict]
```

Query multiple domains and return binary credibility predictions.

**Args:**
- `domains` (list[str]): List of domain names.
- `order` (str): Result ordering. Options: `"original"` (input order) or `"ranked"` (credible=True first, then sorted alphabetically by domain). Default: `"original"`.
- `timeout` (int): Request timeout in seconds. Default: 10.

**Returns:**
- `list[dict]`: List of `{"domain": str, "credible": bool}`. Duplicates are removed; results may be fewer than inputs.

**Raises:**
- `ValueError`: If any domain format is invalid.
- `requests.exceptions.Timeout`: If request exceeds timeout.
- `requests.RequestException`: For HTTP or transport errors.

**Example:**
```python
from credigraph import query_batch

domains = ["apnews.com", "example.com", "bbc.com"]
results = query_batch(domains, order="ranked")
for r in results:
    status = "✓" if r["credible"] else "✗"
    print(f"{status} {r['domain']}")
```

---

## Advanced Queries

These internal APIs provide additional prediction modes beyond binary classification.

### Continuous Score Prediction

Query continuous credibility scores (range: -1 to 1, higher = more credible).

```python
from credigraph import _query_cts, _query_cts_batch

# Single domain
result = _query_cts("apnews.com")
# {"domain": "apnews.com", "credibility_level": 0.85}

# Batch
results = _query_cts_batch(["apnews.com", "example.com"], order="ranked")
# Returns sorted by credibility_level descending
```

### DomainRel Labels

Query ground-truth DomainRel binary labels (external dataset).

```python
from credigraph import _query_domainrel, _query_domainrel_batch

result = _query_domainrel("apnews.com")
# {"domain": "apnews.com", "credible": True}
```

### DQR Regression Labels

Query ground-truth DQR (pc1) regression scores for evaluation.

```python
from credigraph import _query_dqr, _query_dqr_batch

result = _query_dqr("apnews.com")
# {"domain": "apnews.com", "credibility_level": 0.92}
```

---

## Client Class

For advanced use or connection reuse, instantiate `CrediGraphClient` directly.

```python
from credigraph import CrediGraphClient

client = CrediGraphClient(timeout=15)
result = client.query("apnews.com")
```

**Methods:** Same as module-level functions (`query`, `query_batch`, `_query_cts`, etc.)

---

## Domain Normalization

Domains are automatically normalized:
- Lowercase conversion
- URL parsing (http/https prefixes optional)
- "www." prefix removal
- Duplicate removal in batch queries

Valid formats:
```
"apnews.com"
"https://apnews.com"
"http://www.apnews.com"
```

Invalid domains raise `ValueError`.

---

## Error Handling

```python
import requests
from credigraph import query, CrediGraphClient

try:
    result = query("invalid domain!")
except ValueError as e:
    print(f"Invalid domain: {e}")

try:
    result = query("example.com", timeout=1)
except requests.exceptions.Timeout:
    print("Request timed out")

try:
    result = query("example.com")
except requests.RequestException as e:
    print(f"Network error: {e}")
```

---

## Metadata & Health Checks

```python
from credigraph import CrediGraphClient

client = CrediGraphClient()

# Health check
response = requests.post(f"{client.api_url}/health")

# Metadata
response = requests.post(f"{client.api_url}/metadata")
# {"api_version": "0.4.1", "data_cutoff_month": "2024-12", ...}
```

---

## Return Type Reference

### Binary Result
```python
{
    "domain": str,      # Normalized domain name
    "credible": bool    # True if credible, False if not credible
}
```

### Continuous Result
```python
{
    "domain": str,               # Normalized domain name
    "credibility_level": float   # Score (range -1 to 1), rounded to 2 decimals
}
```

### Not Found
If a domain is not in the dataset, HTTP 404 is raised. Batch queries return results only for found domains; unmatched domains are skipped.

---

## See Also

- [CrediGraph Paper](https://credi-net.github.io) - Research and methodology
- [API Endpoints](../openapi.yaml) - Full OpenAPI specification
