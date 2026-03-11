<div align="center">

# CrediNet

<img src="img/credinet.png" alt="CrediNet Logo" style="width: 100px; height: auto;" />

Domain credibility scoring client for fact-checking and online retrieval.

</div>

---

## Install

```bash
pip install credigraph
```

## Quick Start

```python
from credigraph import help, label_sets, metadata, months, query, stats

# Single domain
result = query("apnews.com")
print(result["continuous_score"])
print(result["binary_score"])

# Multiple domains
results = query(["apnews.com", "cnn.com", "reuters.com"])
for result in results:
    print(result["domain"], result["continuous_score"], result["binary_score"])

# API/model metadata
print(metadata())

# Monthly graph stats and downloadable month catalog
print(stats())
print(months())

# Label-set metadata
print(label_sets())

# Endpoint guide
print(help())
```

### Automatic Normalization

The client handles various URL/domain formats:
```python
query("example.com")
query("www.example.com")
query("https://example.com/article?x=1")
query("EXAMPLE.COM")  # case-insensitive
```

It also transparently retries both canonical and flipped domain layouts when the
server-side dataset uses a different storage convention.

### Response Format

```json
{
    "domain": "com.apnews",
    "continuous_score": 0.7,
    "binary_score": 0.91
}
```

### API Metadata

```python
from credigraph import summarize

meta = summarize()
print(meta["api_version"])       # e.g. 0.2.1
print(meta["data_cutoff_month"]) # 2024-12
print(meta["method"])            # GAT-TEXT
print(meta["score_sources"]["regression"])
print(meta["score_sources"]["binary"])
```

### Monthly Graph Endpoints

```python
graph_stats = stats()
print(graph_stats["months"][0]["month"])
print(graph_stats["months"][0]["nodes"])
print(graph_stats["months"][0]["edges"])

catalog = months()
print(catalog["months"][0]["download_url"])
```

### Label-Set Metadata

```python
labels = label_sets()
print(labels["label_sets"][0]["name"])
print(labels["label_sets"][0]["source_url"])
```

### Endpoint Guide

```python
guide = help()
print(guide["api_url"])
print(guide["source"])  # openapi or fallback
print(guide["endpoints"][0]["path"])
```

## Configuration

### Token


```bash
export HF_TOKEN=hf_your_token_here
```
<!-- 
### Custom API Endpoint

```python
from credigraph import CrediGraphClient

client = CrediGraphClient(
    api_url="http://localhost:7860",  # Dev server
    timeout=30
)
result = client.query("example.com")
``` -->

### Environment Variables

<!-- ```bash
export CREDI_API_URL=https://custom-api.example.com
``` -->

```python
from credigraph import CrediGraphClient

client = CrediGraphClient()  # Reads HF_TOKEN automatically
```

## Versioning

This package follows [semantic versioning](https://semver.org/):

```python
import credigraph
print(credigraph.__version__)  # e.g., "0.2.1"
```

## API Contract

- OpenAPI spec: [openapi.yaml](openapi.yaml)
- Developing guide (versioning, testing): [credigraph/README.md](credigraph/README.md)

## Support

- **Issues**: [GitHub Issues](https://github.com/credi-net/CrediNet/issues)
- **Documentation**: this README (client usage) + [credigraph/README.md](credigraph/README.md) (developer workflow)
