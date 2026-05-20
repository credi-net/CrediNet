# CrediGraph API

Query domain-level credibility predictions for fact-checking and retrieval tasks.

**API Version:** 0.4.1  
**Data Cutoff:** December 2024  
**Method:** Content + topology-based

---

## Overview

CrediGraph provides a Python client and REST API for querying domain credibility predictions. Query single domains or batch multiple domains with:

- **Binary classification**: Credible (True/False)
- **Continuous scoring**: Credibility level (0 to 1)
- **Ground-truth labels**: DomainRel and DQR datasets

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
print(result)  # {"domain": "apnews.com", "credible": True}

# Multiple domains
results = query_batch(["apnews.com", "cnn.com", "reuters.com"])
for r in results:
    print(f"{r['domain']}: {r['credible']}")
```

---

## Documentation

- **[Domain Query](domain-query.md)** — Detailed API reference with all functions and return types
- **[API Configuration](api-configuration.md)** — OpenAPI specification, testing, and support
- **[Leaderboard](leaderboard.md)** — CrediBench benchmark results
- **[Versions](versions.md)** — Release history and changelog

