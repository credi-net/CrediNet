from .client import (
    CrediGraphClient,
    _query_cts,
    _query_cts_batch,
    _query_domainrel,
    _query_domainrel_batch,
    _query_dqr,
    _query_dqr_batch,
    query,
    query_batch,
)

__version__ = "0.4.0"
__author__ = "Complex Data Lab"

__all__ = [
    "query",
    "query_batch",
    "_query_cts",
    "_query_cts_batch",
    "_query_domainrel",
    "_query_domainrel_batch",
    "_query_dqr",
    "_query_dqr_batch",
    "CrediGraphClient",
    "__version__",
]
