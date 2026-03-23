# CrediGraph API

Domain credibility scoring client for fact-checking and online retrieval.

## Overview

CrediGraph provides a simple public API for querying domain-level credibility predictions.

The query endpoints are:

- `query(domain)` for one domain
- `query_batch(domains)` for multiple domains

For detailed function signatures and examples, see the [Domain Query Page](domain-query.md). 

For API contract references and support information, see [API Configuration](api-configuration.md).

For release history and public version changes, see [Versions](versions.md).


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
    print(result["domain"], result["credible"])
```

See the [Domain Query Page](domain-query.md) for function signatures, return types, and usage examples.

