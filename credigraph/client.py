import os
import requests
from typing import Any, Dict, List
from credigraph.utils import normalize_domain, normalize_domains

DEFAULT_API_URL = "https://credi-net-credinet.hf.space"


class CrediGraphClient:
    """
    Python client for the CrediGraph API.
    
    Supports client version tracking via User-Agent header.
    
    Args:
        api_url: Base URL of the CrediGraph API (default: production)
        token: HuggingFace API token (or HF_TOKEN environment variable)
        timeout: Request timeout in seconds (default: 10)
    """
    
    def __init__(
        self,
        api_url: str | None = None,
        token: str | None = None,
        timeout: int = 10,
    ):
        self.api_url = api_url or os.getenv("CREDI_API_URL") or DEFAULT_API_URL
        self.token = token or os.getenv("HF_TOKEN")
        self.timeout = timeout
        
        try:
            from . import __version__
            version = __version__
        except ImportError:
            version = "unknown"
        self._user_agent = f"credigraph/{version}"

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers with versioning info."""
        headers = {
            "User-Agent": self._user_agent,
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def query_domain(self, domain: str) -> Dict[str, Any]:
        """
        Query a single domain.
        
        Args:
            domain: Domain name to query
            
        Returns:
            Dictionary with credibility scores and metadata
            
        Raises:
            ValueError: If domain normalization fails
            requests.RequestException: On network or API errors
        """
        domain = normalize_domain(domain)
        url = f"{self.api_url}/by_domain/{domain}"
        headers = self._get_headers()

        try:
            r = requests.get(url, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            result = r.json()
            result["continuous_score"] = result.pop("pc1_score")
            
            if "continuous_score" in result and isinstance(result["continuous_score"], (int, float)):
                result["continuous_score"] = float(f"{result['continuous_score']:.2f}")
            
            return result
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(
                f"Request to {url} timed out after {self.timeout}s"
            )

    def query(self, domains: List[str] | str) -> Dict[str, Any] | List[Dict[str, Any]]:
        """
        Query one or more domains.
        
        Args:
            domains: Single domain string or list of domain strings
            
        Returns:
            Single result dict (if input was string) or list of result dicts
        """
        if isinstance(domains, str):
            return self.query_domain(domains)
        
        domains = normalize_domains(domains)
        return [self.query_domain(d) for d in domains]
