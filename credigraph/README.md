# CrediGraph Developer Guide

This README is for maintainers and contributors.

## Local Setup

```bash
git clone https://github.com/credi-net/CrediNet.git
cd CrediNet
uv sync --active
source .venv/bin/activate
```

## Docs

- End-user docs live in [../README.md](../README.md)
- API contract lives in [../openapi.yaml](../openapi.yaml)
- Versioning and testing workflow lives in [../README.md](../README.md)

## Dev Checks

### Unit tests
```bash
pytest credigraph/test/ -v
```

### API client integration test
```bash
python tests/test_api.py
```

### Contract tests (OpenAPI vs deployed API)
```bash
schemathesis run openapi.yaml -u https://credi-net-credinet.hf.space --seed=42
```

### One-command full check
```bash
./tests/test_all.sh
```

## Release Flow

### 1) Validate before release
```bash
./tests/test_all.sh
```

### 2) Bump version
```bash
# commit to git, and
bump2version patch   # bug fixes
bump2version minor   # backward-compatible features
bump2version major   # breaking changes
```

### 3) Build + publish package
```bash
rm -rf dist/
python -m build
twine upload dist/*
```

## Internal Methods

Access internal query methods for specialized use cases:

- **Continuous predictions**: `_query_cts()`, `_query_cts_batch()` — Model regression scores
- **DomainRel labels**: `_query_domainrel()`, `_query_domainrel_batch()` — Ground-truth binary labels
- **DQR labels**: `_query_dqr()`, `_query_dqr_batch()` — Ground-truth regression labels

Example:

```python
from credigraph import _query_dqr, _query_domainrel, _query_cts

# DQR regression labels (ground-truth pc1)
dqr_result = _query_dqr("apnews.com")
# {"domain": "apnews.com", "credibility_level": 0.77}

# DomainRel binary labels (ground-truth bin)
dr_result = _query_domainrel("apnews.com")
# {"domain": "apnews.com", "credible": true}

# Continuous predictions (model regression)
cts_result = _query_cts("apnews.com")
# {"domain": "apnews.com", "credibility_level": 0.85}

# Batch queries with ranking
dqr_results = _query_dqr_batch(["example.com", "apnews.com"], order="ranked")
# Ranked by credibility_level descending
```

## Versioning 

Walkthrough with OpenAPI spec: 

1. **Validate spec:** 
  ```bash
  pip install openapi-spec-validator
  openapi-spec-validator openapi.yaml
  ``` 

<!-- 1.1 ***Visualize:*** (optional) -->
2. **Contract testing:**
  `Schemathesis` reads the spec, generates random valid inputs and tests the API against them to catch mismatches: 
  ```bash
  schemathesis run openapi.yaml \
    --url=https://credi-net-credinet.hf.space \
    --max-examples=100 \
  ```

3. **Test python client against API:**
  ```bash
  python tests/test_api.py
  ```
