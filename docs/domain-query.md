# Domain Query

## Overview

The public interface consists of:

- `query(domain: str) -> DomainResult`
- `query_batch(domains: list[str]) -> list[DomainResult]`

## Return Type

```python
from typing import TypedDict


class DomainResult(TypedDict):
    domain: str
    credible: bool
```

`credible` is the binary credibility prediction for the given domain.

## `query`

```python
def query(
    domain: str,
    timeout: int = 10,
) -> DomainResult: ...
```

Query one domain.

Parameters:

- `domain`: Domain name to query.
- `timeout`: Request timeout in seconds.

Returns:

- `DomainResult`

Example:

```python
from credigraph import query

result = query("apnews.com")

# then, 
result == {
    "domain": "apnews.com",
    "credible": True,
}
```

## `query_batch`

```python
def query_batch(
    domains: list[str],
    timeout: int = 10,
) -> list[DomainResult]: ...
```

Query multiple domains in one call.

Parameters:

- `domains`: List of domain names to query.
- `timeout`: Request timeout in seconds.

Returns:

- `list[DomainResult]`

Example:

```python
from credigraph import query_batch

results = query_batch(["apnews.com", "cnn.com", "reuters.com"])
for result in results:

# then,
results == [
    {"domain": "apnews.com", "credible": True},
    {"domain": "cnn.com", "credible": False},
    {"domain": "reuters.com", "credible": False},
]
```

## Errors

- `requests.exceptions.Timeout` when the request exceeds the configured timeout.
- `requests.RequestException` for HTTP or transport errors.
