# CrediGraph

Domain credibility scoring client for fact-checking and online retrieval.

[docs in progress]

 
## Overview

* **`CrediGraph API`:** query credibility scores from 45M+ domains based on our content- and topology-based methodology. These predictions are made by GNNs trained webgraphs sourced from [Common Crawl](https://commoncrawl.org/web-graphs) and content scraped on the web (*see also* our datasets below). 
* **Datasets:** find our available datasets on [Huggingface](https://huggingface.co/credi-net).
* **TAG Construction Pipeline:** our optimized graph construction pipeline structures Common Crawl and web-scraped data into Text-Attributed temporal Graphs (TAGs) is available and ready to use on GitHub ([Graph](https://github.com/credi-net/CrediGraph), [Text](https://github.com/credi-net/CrediText)). 


# Quick Overview

## Install

```bash
pip install credigraph
```

## Quick Start

```python
from credigraph import query, query_batch

# Single domain
result = query("apnews.com")
print(result["credible"])  # True or False

# Multiple domains
results = query_batch(["apnews.com", "cnn.com", "reuters.com"])
for result in results:
    print(result["credible"])  # True or False for each
```

## Getting Predictions

### `query(domain: str) -> dict`
Queries the credibility of a single domain.
- **Parameters**: `domain` (str) - The domain to check.
- **Returns**: `{"credible": bool}` - True if reliable, False if unreliable.

### `query_batch(domains: list[str], order: str = "original") -> list[dict]`
Queries the credibility of multiple domains in batch.
- **Parameters**: 
  - `domains` (list[str]) - List of domains to check.
  - `order` (str) - "original" (default, same order as input) or "ranked" (sorted by credible True first, then False, within each group alphabetical by domain).
- **Returns**: List of `{"credible": bool}` for each domain.

### `query_internal(domain: str) -> dict`
Queries a single domain and returns both continuous credibility and binary label.
- **Parameters**: `domain` (str) - The domain to check.
- **Returns**: `{"domain": str, "credibility_level": float, "credible": bool}`.

### `query_internal_batch(domains: list[str], order: str = "original") -> list[dict]`
Queries multiple domains and returns internal results for each domain.
- **Parameters**: 
  - `domains` (list[str]) - List of domains to check.
  - `order` (str) - "original" (default, same order as input) or "ranked" (sorted by credibility_level descending, then domain ascending).
- **Returns**: List of `{ "domain": str, "credibility_level": float, "credible": bool }` for each domain.

