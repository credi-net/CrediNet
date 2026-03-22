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

## API Field Naming Convention

- Canonical score field: `credible : boolean` with 0 = not credible, 1 = credible.
- Continuous field: `credibility_level : float` in [0,1].

```yaml
credibility_level:
  type: number
  minimum: 0
  maximum: 1
  description: Continuous credibility score (0-1), rounded to 2 decimal places

credible:
  type: boolean
  description: Binary credibility classification 
```

Modifying response fields:
1. [../openapi.yaml](../openapi.yaml)
2. API backend response
3. `tests/test_api.py`
4. Root [../README.md](../README.md) if user-facing behavior changed


## Internal API & Client Access

For detailed credibility scores (continuous + binary), use the internal client methods with a team token.

[PENDING: internal client methods will soon integrate human-labelled sets as well]

### Token Mechanism

Use internal token as follows: 
```bash
export CREDI_INTERNAL_TOKEN="token"
```
in shell, or giving it directly to the client:  
```python
from credigraph import CrediGraphClient

client = CrediGraphClient(token="token")

result = client.query_internal("apnews.com")
print(result)
# Output: {"domain": "apnews.com", "credibility_level": 0.85, "credible": True}

results = client.query_internal_batch(["apnews.com", "cnn.com"])
# Sort by credibility_level
results = client.query_internal_batch(["example.com", "apnews.com"], order="ranked")
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
