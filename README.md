<div align="center">

# CrediNet

<img src="img/credinet.png" alt="CrediNet Logo" style="width: 100px; height: auto;" />

Graph and Machine Learning-based domain credibility assessments on the web.

[![PyPI](https://img.shields.io/pypi/v/credigraph?style=flat&label=PyPI&labelColor=white&logo=pypi&logoColor=black)](https://pypi.org/project/credigraph/)
[![Downloads](https://img.shields.io/pypi/dm/credigraph?style=flat&label=Downloads&labelColor=white&logo=pypi&logoColor=black)](https://pypi.org/project/credigraph/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.23340)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-CrediNet-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/credi-net)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

For more details, refer to the full documentation:

[![Docs](https://img.shields.io/readthedocs/credinet?style=flat&label=Docs&labelColor=white&logo=readthedocs&logoColor=black)](https://credinet.readthedocs.io/en/latest/) 
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

### Input Normalization

The client handles various input formats:
```python
query("example.com")
query("www.example.com")
query("https://example.com/article?x=1")
query("EXAMPLE.COM") 
# will all resolve to example.com
```

### Response Format

For one domain's credibility score, use the `query()` endpoint:

```json
{
    "domain": "apnews.com",
    "credible": true
}
```

For a list of domains, use the `query_batch()` batch endpoint:

```json
[
    {"domain": "apnews.com", "credible": true},
    {"domain": "cnn.com", "credible": true},
    {"domain": "example.com", "credible": false}
]
```

Refer to the [full documentation](https://credinet.readthedocs.io/en/latest/domain-query/) for more details. 


## Versioning

The CrediGraph API follows [semantic versioning](https://semver.org/):

```python
import credigraph
print(credigraph.__version__)  # 0.3.1
```

Refer to the [full versioning documentation](https://credinet.readthedocs.io/en/latest/versions/) for more details. 

## API Contract

- OpenAPI spec: [openapi.yaml](openapi.yaml)
- Developing guide (versioning, testing): [credigraph/README.md](credigraph/README.md)

Refer to the [full config documentation](https://credinet.readthedocs.io/en/latest/api-configuration/) for more details. 

## Support

- **Issues**: [GitHub Issues](https://github.com/credi-net/CrediNet/issues)
- **Documentation**: read the [official CrediGraph documentation](https://credinet.readthedocs.io/en/latest/) or the repo's [README](README.md) (client usage) + [credigraph/README.md](credigraph/README.md) (developer workflow)
- **Further issues:** [contact the developers](emailto:emma.kondrup@mila.quebec).
