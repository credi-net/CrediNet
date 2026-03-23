from .client import (
    CrediGraphClient,
    query,
    query_batch,
    query_internal,
    query_internal_batch,
    stats,
    months,
)

__version__ = "0.3.2"
__author__ = "Complex Data Lab"

__all__ = [
    "query",
    "query_batch",
    "query_internal",
    "query_internal_batch",
    "stats",
    "months",
    "CrediGraphClient",
    "__version__",
]
