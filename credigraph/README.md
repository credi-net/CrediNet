# CrediGraph Developer Guide

This README is for maintainers and contributors (internal team).

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
- Versioning and testing workflow is documented in this guide and [../openapi.yaml](../openapi.yaml)

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
# commit to git, then
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

## API Field Naming Convention

- Canonical score field is `continuous_score`
- Canonical binary field is `binary_score`
- API metadata endpoint: `GET /metadata`
- Label-set metadata endpoint: `GET /label_sets`

```yaml
binary_score:
  type: number
  minimum: 0
  maximum: 1
  description: Binary credibility score in {0,1}
```

When introducing new response fields, update in this order:
1. [../openapi.yaml](../openapi.yaml)
2. API backend response
3. `tests/test_api.py`
4. Root [../README.md](../README.md) if user-facing behavior changed


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
  <!-- # Expected output summary:
  # GET /by_domain/{domain} .. 100 passed
  # GET /health .............. 50 passed
  # Total: 150 passed, 0 failed
  ``` -->

3. **Test python client against API:**
  ```bash
  python tests/test_api.py
  ```
