"""CrediGraph API Client"""

from .client import CrediGraphClient

__version__ = "0.3.0"
__author__ = "Complex Data Lab"

__all__ = [
    "CrediGraphClient",
    "help",
    "health",
    "query",
    "metadata",
    "summarize",
    "stats",
    "months",
    "label_sets",
    "__version__",
]


def query(
    domain: str,
    api_url: str | None = None,
    token: str | None = None,
    timeout: int = 10,
    raw: bool = False,
    **kwargs
):
    """
    Convenience function to query a domain using default client settings.
    
    Args:
        domain: Single domain or list of domains to query
        api_url: Optional custom API endpoint
        token: HF token (or use HF_TOKEN environment variable)
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
    return client.query(domain, raw=raw)


def health(
    api_url: str | None = None,
    token: str | None = None,
    timeout: int = 10,
    **kwargs
):
    """Convenience function to fetch API health information."""
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        timeout=timeout,
        **kwargs
    )
    return client.health()


def metadata(
    api_url: str | None = None,
    token: str | None = None,
    timeout: int = 10,
    **kwargs
):
    """Convenience function to fetch API metadata."""
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        timeout=timeout,
        **kwargs
    )
    return client.metadata()


def summarize(
    api_url: str | None = None,
    token: str | None = None,
    timeout: int = 10,
    **kwargs
):
    """Alias for metadata()."""
    return metadata(api_url=api_url, token=token, timeout=timeout, **kwargs)


def stats(
    api_url: str | None = None,
    token: str | None = None,
    internal_token: str | None = None,
    timeout: int = 10,
    **kwargs
):
    """Convenience function to fetch monthly graph composition statistics."""
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        internal_token=internal_token,
        timeout=timeout,
        **kwargs
    )
    return client.stats()


def months(
    api_url: str | None = None,
    token: str | None = None,
    internal_token: str | None = None,
    timeout: int = 10,
    **kwargs
):
    """Convenience function to fetch downloadable monthly dataset entries."""
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        internal_token=internal_token,
        timeout=timeout,
        **kwargs
    )
    return client.months()


def label_sets(
    api_url: str | None = None,
    token: str | None = None,
    internal_token: str | None = None,
    timeout: int = 10,
    **kwargs
):
    """Convenience function to fetch label-set metadata."""
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        internal_token=internal_token,
        timeout=timeout,
        **kwargs
    )
    return client.label_sets()


def help(
    api_url: str | None = None,
    token: str | None = None,
    timeout: int = 10,
    raw: bool = False,
    **kwargs
):
    """Return a user-friendly API guide (or raw payload when raw=True)."""
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        timeout=timeout,
        **kwargs
    )
    payload = client.help()
    if raw:
        return payload

    lines = [
        "CrediGraph API Help",
        f"Base URL: {payload.get('api_url', 'unknown')}",
        f"Client: {payload.get('client_user_agent', 'unknown')}",
    ]

    if payload.get("api_version"):
        lines.append(f"API version: {payload['api_version']}")
    if payload.get("method"):
        lines.append(f"Model method: {payload['method']}")
    if payload.get("data_cutoff_month"):
        lines.append(f"Data cutoff month: {payload['data_cutoff_month']}")

    lines.append("")
    lines.append("Commands:")
    lines.append("  credigraph.help()  # formatted API guide as text")
    lines.append("  credigraph.help(raw=True)  # raw machine-readable guide dict")
    lines.append("  credigraph.health()  # {'status', 'api_version'}")
    lines.append("  credigraph.metadata()  # {'api_version', 'data_cutoff_month', 'method', 'score_sources'}")
    lines.append("  credigraph.query('apnews.com', raw=False)  # optional raw output for full float result")
    lines.append("  credigraph.query(['apnews.com', 'cnn.com'], raw=False)  # optional raw output for full float result")
    lines.append("")
    lines.append(f"Source: {payload.get('source', 'unknown')}")

    return "\n".join(lines)


__all__ = [
    "help",
    "health",
    "query",
    "metadata",
    "summarize",
    "stats",
    "months",
    "label_sets",
    "CrediGraphClient",
    "__version__",
]