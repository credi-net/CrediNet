"""CrediGraph API Client"""

from .client import CrediGraphClient

__version__ = "0.1.4"
__author__ = "Complex Data Lab"

def query(domain: str, **kwargs):
    return CrediGraphClient(**kwargs).query(domain)

__all__ = ["query", "CrediGraphClient", "__version__"]