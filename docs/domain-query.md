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


class InternalDomainResult(TypedDict, total=False):
    domain: str
    credible: bool
    credibility_level: float
```

`credible` is the binary credibility prediction returned by the API.

For internal methods:

- `_query_cts*` returns continuous regression predictions from model.
- `_query_domainrel*` returns binary labels from DomainRel ground-truth.
- `_query_dqr*` returns continuous regression labels from DQR ground-truth.

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

## Internal Methods

### `_query_cts`

Query continuous (regression) credibility predictions.

```python
def _query_cts(domain: str) -> InternalDomainResult: ...
def _query_cts_batch(domains: list[str], order: str = "original") -> list[InternalDomainResult]: ...
```

Example:

```python
from credigraph import _query_cts, _query_cts_batch

result = _query_cts("apnews.com")
# Returns: {"domain": "apnews.com", "credibility_level": 0.85}

results = _query_cts_batch(["apnews.com", "cnn.com"], order="ranked")
# Ranked by credibility_level descending
```

### `_query_domainrel`

Query binary credibility from DomainRel ground-truth labels.

```python
def _query_domainrel(domain: str) -> DomainResult: ...
def _query_domainrel_batch(domains: list[str], order: str = "original") -> list[DomainResult]: ...
```

Example:

```python
from credigraph import _query_domainrel, _query_domainrel_batch

result = _query_domainrel("apnews.com")
# Returns: {"domain": "apnews.com", "credible": true}

results = _query_domainrel_batch(["apnews.com", "cnn.com"], order="ranked")
# Ranked by credible (True first)
```

### `_query_dqr`

Query regression credibility from DQR ground-truth labels.

```python
def _query_dqr(domain: str) -> InternalDomainResult: ...
def _query_dqr_batch(domains: list[str], order: str = "original") -> list[InternalDomainResult]: ...
```

Example:

```python
from credigraph import _query_dqr, _query_dqr_batch

result = _query_dqr("apnews.com")
# Returns: {"domain": "apnews.com", "credibility_level": 0.77}

results = _query_dqr_batch(["apnews.com", "cnn.com"], order="ranked")
# Ranked by credibility_level descending
```

## Errors

- `ValueError` on invalid domain input.
- `requests.exceptions.Timeout` when the request exceeds the configured timeout.
- `requests.RequestException` for HTTP or transport errors.
