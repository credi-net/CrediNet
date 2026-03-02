"""CrediGraph API Client"""

from .client import CrediGraphClient

__version__ = "0.1.5"
__author__ = "Complex Data Lab"

__all__ = ["CrediGraphClient", "query", "__version__"]


def query(
    domain: str,
    api_url: str | None = None,
    token: str | None = None,
    timeout: int = 10,
    **kwargs
):
    """
    Convenience function to query a domain using default client settings.
    
    Args:
        domain: Single domain or list of domains to query
        api_url: Optional custom API endpoint
        token: Optional HF token (or use HF_TOKEN environment variable)
        timeout: Request timeout in seconds (default: 10)
        **kwargs: Additional args passed to CrediGraphClient
        
    Returns:
        Domain credibility data from the API
    """
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        timeout=timeout,
        **kwargs
    )
    return client.query(domain)


__all__ = ["query", "CrediGraphClient", "__version__"]