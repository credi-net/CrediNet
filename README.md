<div align="center">

# CrediNet

<img src="img/credinet.png" alt="CrediNet Logo" style="width: 100px; height: auto;" />

Domain credibility scoring client for fact-checking and online retrieval.

[![PyPI](https://img.shields.io/pypi/v/credigraph?style=flat&label=PyPI&labelColor=white&logo=pypi&logoColor=black)](https://pypi.org/project/credigraph/)
[![Downloads](https://img.shields.io/pypi/dm/credigraph?style=flat&label=Downloads&labelColor=white&logo=pypi&logoColor=black)](https://pypi.org/project/credigraph/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.23340)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-CrediNet-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/credi-net)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

<!-- TODO: [![Docs](https://img.shields.io/readthedocs/credigraph?style=flat&label=Docs&labelColor=white&logo=readthedocs&logoColor=black)](https://credigraph.readthedocs.io/) --> 
</div>

---

## Install

```bash
pip install credigraph
```

## Quick Start

```python
from credigraph import query, query_batch

# Single domain
result = query("apnews.com")
print(result["credible"])

# Multiple domains
results = query_batch(["apnews.com", "cnn.com", "reuters.com"])
for result in results:
    print(result["credible"])
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

For the `query()` binary endpoint:

```json
{
    "domain": "apnews.com",
    "credible": true
}
```

For the `query_batch()` binary batch endpoint:

```json
[
    {"domain": "apnews.com", "credible": true},
    {"domain": "cnn.com", "credible": true},
    {"domain": "example.com", "credible": false}
]
```

## Versioning

This package follows [semantic versioning](https://semver.org/):

```python
import credigraph
print(credigraph.__version__)  # 0.3.1
```

## API Contract

- OpenAPI spec: [openapi.yaml](openapi.yaml)
- Developing guide (versioning, testing): [credigraph/README.md](credigraph/README.md)

## Support

- **Issues**: [GitHub Issues](https://github.com/credi-net/CrediNet/issues)
- **Documentation**: this README (client usage) + [credigraph/README.md](credigraph/README.md) (developer workflow)
- **Further issues:** [contact repo developer](emailto:emma.kondrup@mila.quebec)
