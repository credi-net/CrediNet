from .client import (
    CrediGraphClient,
    months,
    query,
    query_batch,
    query_internal,
    query_internal_batch,
    stats,
)

__version__ = "0.3.4"
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
