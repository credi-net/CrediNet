import os
import re
import requests
from typing import Any, Dict, List
from urllib.parse import urlparse

DEFAULT_API_URL = "https://credi-net-credinet.hf.space"
DOMAIN_REGEX = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,}$")


def _canonicalize_domain(value: str) -> str:
    """Normalize user domain input to canonical host form (not flipped)."""
    if not value or not isinstance(value, str):
        raise ValueError("Domain must be a non-empty string")

    value = value.strip().lower()
    if not value.startswith(("http://", "https://")):
        value = "http://" + value

    parsed = urlparse(value)
    host = parsed.hostname or ""
    if host.startswith("www."):
        host = host[4:]

    if not DOMAIN_REGEX.match(host):
        raise ValueError(f"Invalid domain: {host}")
    return host


def _canonicalize_domains(domains: List[str]) -> List[str]:
    seen = set()
    clean = []
    for d in domains:
        nd = _canonicalize_domain(d)
        if nd not in seen:
            seen.add(nd)
            clean.append(nd)
    return clean


class CrediGraphClient:
    """
    Python client for the CrediGraph API.
        
    Args:
        api_url: Base URL of the CrediGraph API (default: production)
        token: Token for internal API access (for supplemental )
        timeout: Request timeout in seconds (default: 10)
    """
    
    def __init__(
        self,
        api_url: str | None = None,
        token: str | None = None,
        timeout: int = 10,
    ):
        self.api_url = api_url or os.getenv("CREDI_API_URL") or DEFAULT_API_URL
        self.single_q_url = f"{self.api_url}/by_domain"
        self.batch_q_url = f"{self.api_url}/by_domains"
        self.token = token or os.getenv("CREDI_INTERNAL_TOKEN")
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
        return headers

    def query(self, domain: str) -> Dict[str, Any]:
        """
        Query a single domain and return domain + binary credibility.

        Args:
            domain: Single domain string

        Returns:
            Dictionary with domain and credible bool
        """
        canonical = _canonicalize_domain(domain)
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"

        try:
            r = requests.post(self.single_q_url, params={"domain": canonical}, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            result = r.json()

            return {
                "domain": result["domain"],
                "credible": result["credible"],
            }
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(
                f"Request to {self.single_q_url} timed out after {self.timeout}s"
            )

    def query_batch(self, domains: List[str], order: str = "original") -> List[Dict[str, Any]]:
        """
        Query multiple domains and return list of domain + binary credible responses.

        Args:
            domains: List of domain strings
            order: "original" (default) or "ranked" (sort by credible True first, then False, within each alphabetical by domain)

        Returns:
            List of dictionaries with domain and credible bool
        """
        canonical_domains = _canonicalize_domains(domains)
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"

        try:
            r = requests.post(self.batch_q_url, json=canonical_domains, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            results = r.json()

            processed = []
            for item in results:
                out = {"domain": item["domain"]}
                if "credible" in item:
                    out["credible"] = item["credible"]
                processed.append(out)

            if order == "ranked":
                # Sort by credible (True first), then domain alphabetical.
                processed.sort(
                    key=lambda x: (
                        -(1 if x.get("credible") is True else 0),
                        x["domain"],
                    )
                )

            return processed
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(
                f"Request to {self.batch_q_url} timed out after {self.timeout}s"
            )

    def query_internal(self, domain: str) -> Dict[str, Any]:
        """
        [Internal] Query a single domain and return both binary and continuous credibility.
        
        Args:
            domain: Single domain string

        Returns:
            Dictionary with domain, credibility_level, and credible fields
            
        Raises:
            PermissionError: If internal_token not provided
        """
        if not self.token:
            raise PermissionError(
                "token required for query_internal(). "
                "Pass to CrediGraphClient(token=...) or set CREDI_INTERNAL_TOKEN env var."
            )
        
        canonical = _canonicalize_domain(domain)
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        headers["X-Internal-Token"] = self.token

        try:
            r = requests.post(self.single_q_url, params={"domain": canonical}, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            result = r.json()

            return {
                "domain": result["domain"],
                "credibility_level": result["credibility_level"],
                "credible": result["credible"],
            }
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(
                f"Request to {self.single_q_url} timed out after {self.timeout}s"
            )

    def query_internal_batch(self, domains: List[str], order: str = "original") -> List[Dict[str, Any]]:
        """
        [Internal] Query multiple domains and return list of internal credibility results.
        
        Args:
            domains: List of domain strings
            order: "original" (default) or "ranked" (sort by credibility_level descending, then domain ascending)

        Returns:
            List of dicts with domain, credibility_level, and credible fields
            
        Raises:
            PermissionError: If internal_token not provided
        """
        if not self.token:
            raise PermissionError(
                "token required for query_internal_batch(). "
                "Pass to CrediGraphClient(token=...) or set CREDI_INTERNAL_TOKEN env var."
            )
        
        canonical_domains = _canonicalize_domains(domains)
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        headers["X-Internal-Token"] = self.token

        try:
            r = requests.post(self.batch_q_url, json=canonical_domains, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            results_data = r.json()

            results = []
            for item in results_data:
                out = {"domain": item["domain"]}
                if "credibility_level" in item:
                    out["credibility_level"] = item["credibility_level"]
                if "credible" in item:
                    out["credible"] = item["credible"]
                results.append(out)

            if order == "ranked":
                # Sort by credibility_level descending, then domain ascending
                results.sort(
                    key=lambda x: (
                        -float(x.get("credibility_level", float("-inf"))),
                        x["domain"],
                    )
                )

            return results
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(
                f"Request to {self.batch_q_url} timed out after {self.timeout}s"
            )

    def stats(self) -> dict:
        """
        [Internal] Get graph statistics.
                
        Returns:
            Dictionary with months array containing stats (nodes, edges, overlaps, etc.)
            
        Raises:
            PermissionError: If internal_token not provided
        """
        if not self.token:
            raise PermissionError(
                "token required for stats(). "
                "Pass to CrediGraphClient(token=...) or set CREDI_INTERNAL_TOKEN env var."
            )
        
        url = f"{self.api_url}/stats"
        headers = self._get_headers()
        headers["X-Internal-Token"] = self.token

        try:
            r = requests.get(url, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(
                f"Request to {url} timed out after {self.timeout}s"
            )

    def months(self) -> dict:
        """
        [Internal] Get available monthly knowledge graph snapshots.
                
        Returns:
            Dictionary with months array containing snapshot metadata
            
        Raises:
            PermissionError: If internal_token not provided
        """
        if not self.token:
            raise PermissionError(
                "token required for months(). "
                "Pass to CrediGraphClient(token=...) or set CREDI_INTERNAL_TOKEN env var."
            )
        
        url = f"{self.api_url}/months"
        headers = self._get_headers()
        headers["X-Internal-Token"] = self.token

        try:
            r = requests.get(url, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(
                f"Request to {url} timed out after {self.timeout}s"
            )


def query(
    domain: str,
    api_url: str | None = None,
    token: str | None = None,
    timeout: int = 10,
):
    """Convenience function to query a single domain (binary credible only)."""
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        timeout=timeout,
    )
    return client.query(domain)


def query_batch(
    domains: list[str],
    order: str = "original",
    api_url: str | None = None,
    token: str | None = None,
    timeout: int = 10,
):
    """Convenience function to query multiple domains (binary credible only)."""
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        timeout=timeout,
    )
    return client.query_batch(domains, order=order)


def query_internal(
    domain: str,
    api_url: str | None = None,
    token: str | None = None,
    timeout: int = 10,
):
    """Convenience function to query a domain with full details. Requires token."""
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        timeout=timeout,
    )
    return client.query_internal(domain)


def query_internal_batch(
    domains: list[str],
    order: str = "original",
    api_url: str | None = None,
    token: str | None = None,
    timeout: int = 10,
):
    """Convenience function to query multiple domains with full details. Requires token."""
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        timeout=timeout,
    )
    return client.query_internal_batch(domains, order=order)


def stats(
    api_url: str | None = None,
    token: str | None = None,
    timeout: int = 10,
):
    """Get graph statistics. Requires token."""
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        timeout=timeout,
    )
    return client.stats()


def months(
    api_url: str | None = None,
    token: str | None = None,
    timeout: int = 10,
):
    """Get available monthly knowledge graph snapshots. Requires token."""
    client = CrediGraphClient(
        api_url=api_url,
        token=token,
        timeout=timeout,
    )
    return client.months()
