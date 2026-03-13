<div align="center">

# CrediNet

<img src="img/credinet.png" alt="CrediNet Logo" style="width: 100px; height: auto;" />

Domain credibility scoring client for fact-checking and online retrieval.

[![PyPI](https://img.shields.io/pypi/v/credigraph?style=flat&label=PyPI&labelColor=white&logo=pypi&logoColor=black)](https://pypi.org/project/credigraph/)
[![Downloads](https://img.shields.io/pypi/dm/credigraph?style=flat&label=Downloads&labelColor=white&logo=pypi&logoColor=black)](https://pypi.org/project/credigraph/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.23340)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-CrediNet-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/credi-net)

<!-- TODO: [![Docs](https://img.shields.io/readthedocs/credigraph?style=flat&label=Docs&labelColor=white&logo=readthedocs&logoColor=black)](https://credigraph.readthedocs.io/) --> 
</div>

---

## Install

```bash
pip install credigraph
```

## Quick Start

```python
from credigraph import query

# Single domain
result = query("apnews.com")
print(result["continuous_score"])

# Multiple domains
results = query(["apnews.com", "cnn.com", "reuters.com"])
for result in results:
    print(result["domain"], result["continuous_score"])
```

### Automatic Normalization

The client handles various URL/domain formats:
```python
query("example.com")
query("www.example.com")
query("https://example.com/article?x=1")
query("EXAMPLE.COM")  # case-insensitive
```

### Response Format

```json
{
    "domain": "com.apnews",
    "continuous_score": 0.7
}
```

## Configuration

### Token (Optional)

The public API works without authentication. Set a token only if your deployment requires it:

```bash
export HF_TOKEN=hf_your_token_here
```

### Custom API Endpoint

```python
from credigraph import CrediGraphClient

client = CrediGraphClient(
    api_url="http://localhost:7860",  # Dev server
    timeout=30
)
result = client.query("example.com")
```

### Environment Variables

```bash
export CREDI_API_URL=https://custom-api.example.com
```

```python
from credigraph import CrediGraphClient

client = CrediGraphClient()  # Reads CREDI_API_URL and HF_TOKEN automatically
```

## Versioning

This package follows [semantic versioning](https://semver.org/):

```python
import credigraph
print(credigraph.__version__)  # e.g., "0.1.5"
```

## API Contract

- OpenAPI spec: [openapi.yaml](openapi.yaml)
- Developing guide (versioning, testing): [credigraph/README.md](credigraph/README.md)

## Support

- **Issues**: [GitHub Issues](https://github.com/credi-net/CrediNet/issues)
- **Documentation**: this README (client usage) + [credigraph/README.md](credigraph/README.md) (developer workflow)
