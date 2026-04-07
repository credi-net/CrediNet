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
        timeout: Request timeout in seconds (default: 10)
    """
    
    def __init__(
        self,
        timeout: int = 10,
    ):
        self.api_url = DEFAULT_API_URL
        self.single_q_url = f"{self.api_url}/by_domain"
        self.batch_q_url = f"{self.api_url}/by_domains"
        self.single_q_cts_url = f"{self.api_url}/by_domain_cts"
        self.batch_q_cts_url = f"{self.api_url}/by_domains_cts"
        self.single_q_domainrel_url = f"{self.api_url}/by_domain_dr"
        self.batch_q_domainrel_url = f"{self.api_url}/by_domains_dr"
        self.single_q_dqr_url = f"{self.api_url}/by_domain_dqr"
        self.batch_q_dqr_url = f"{self.api_url}/by_domains_dqr"
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

    def _post_single(self, url: str, domain: str) -> Dict[str, Any]:
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        try:
            r = requests.post(
                url,
                params={"domain": domain},
                headers=headers,
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(
                f"Request to {url} timed out after {self.timeout}s"
            )

    def _post_batch(self, url: str, domains: List[str]) -> List[Dict[str, Any]]:
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        try:
            r = requests.post(url, json=domains, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(
                f"Request to {url} timed out after {self.timeout}s"
            )

    def query(self, domain: str) -> Dict[str, Any]:
        """
        Query a single domain and return domain + binary credibility.

        Args:
            domain: Single domain string

        Returns:
            Dictionary with domain and credible bool
        """
        canonical = _canonicalize_domain(domain)
        result = self._post_single(self.single_q_url, canonical)

        return {
            "domain": result["domain"],
            "credible": result["credible"],
        }

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
        results = self._post_batch(self.batch_q_url, canonical_domains)

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

    def _query_cts(self, domain: str) -> Dict[str, Any]:
        """
        Query a single domain using prediction continuous score only.

        Args:
            domain: Single domain string

        Returns:
            Dictionary with domain and credibility_level
        """
        canonical = _canonicalize_domain(domain)
        result = self._post_single(self.single_q_cts_url, canonical)

        return {
            "domain": result["domain"],
            "credibility_level": result["credibility_level"],
        }

    def _query_cts_batch(self, domains: List[str], order: str = "original") -> List[Dict[str, Any]]:
        """
        Query multiple domains using prediction continuous score only.

        Args:
            domains: List of domain strings
            order: "original" (default) or "ranked" (sort by credibility_level descending, then domain ascending)

        Returns:
            List of dicts with domain and credibility_level fields
        """
        canonical_domains = _canonicalize_domains(domains)
        results_data = self._post_batch(self.batch_q_cts_url, canonical_domains)

        results = []
        for item in results_data:
            out = {"domain": item["domain"]}
            if "credibility_level" in item:
                out["credibility_level"] = item["credibility_level"]
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

    def _query_domainrel(self, domain: str) -> Dict[str, Any]:
        """
        Query a single domain using DomainRel binary label.

        Args:
            domain: Single domain string

        Returns:
            Dictionary with domain and credible fields
        """
        canonical = _canonicalize_domain(domain)
        result = self._post_single(self.single_q_domainrel_url, canonical)
        return {
            "domain": result["domain"],
            "credible": result["credible"],
        }

    def _query_domainrel_batch(self, domains: List[str], order: str = "original") -> List[Dict[str, Any]]:
        """
        Query multiple domains using DomainRel binary labels.

        Args:
            domains: List of domain strings
            order: "original" (default) or "ranked" (sort by credible True first, then False, within each alphabetical by domain)

        Returns:
            List of dicts with domain and credible fields
        """
        canonical_domains = _canonicalize_domains(domains)
        results_data = self._post_batch(self.batch_q_domainrel_url, canonical_domains)

        results = []
        for item in results_data:
            out = {"domain": item["domain"]}
            if "credible" in item:
                out["credible"] = item["credible"]
            results.append(out)

        if order == "ranked":
            results.sort(
                key=lambda x: (
                    -(1 if x.get("credible") is True else 0),
                    x["domain"],
                )
            )

        return results

    def _query_dqr(self, domain: str) -> Dict[str, Any]:
        """
        Query a single domain using GT regression labels (DQR pc1).

        Args:
            domain: Single domain string

        Returns:
            Dictionary with domain and credibility_level fields
        """
        canonical = _canonicalize_domain(domain)
        result = self._post_single(self.single_q_dqr_url, canonical)

        return {
            "domain": result["domain"],
            "credibility_level": result["credibility_level"],
        }

    def _query_dqr_batch(self, domains: List[str], order: str = "original") -> List[Dict[str, Any]]:
        """
        Query multiple domains using GT regression labels (DQR pc1).

        Args:
            domains: List of domain strings
            order: "original" (default) or "ranked" (sort by credibility_level descending, then domain ascending)

        Returns:
            List of dicts with domain and credibility_level fields
        """
        canonical_domains = _canonicalize_domains(domains)
        results_data = self._post_batch(self.batch_q_dqr_url, canonical_domains)

        results = []
        for item in results_data:
            out = {"domain": item["domain"]}
            if "credibility_level" in item:
                out["credibility_level"] = item["credibility_level"]
            results.append(out)

        if order == "ranked":
            results.sort(
                key=lambda x: (
                    -float(x.get("credibility_level", float("-inf"))),
                    x["domain"],
                )
            )

        return results

def query(
    domain: str,
    timeout: int = 10,
):
    """Convenience function to query a single domain (binary credible only)."""
    client = CrediGraphClient(timeout=timeout)
    return client.query(domain)


def query_batch(
    domains: list[str],
    order: str = "original",
    timeout: int = 10,
):
    """Convenience function to query multiple domains (binary credible only)."""
    client = CrediGraphClient(timeout=timeout)
    return client.query_batch(domains, order=order)


def _query_cts(
    domain: str,
    timeout: int = 10,
):
    """Convenience function to query a domain with continuous score only."""
    client = CrediGraphClient(timeout=timeout)
    return client._query_cts(domain)


def _query_cts_batch(
    domains: list[str],
    order: str = "original",
    timeout: int = 10,
):
    """Convenience function to query domains with continuous score only."""
    client = CrediGraphClient(timeout=timeout)
    return client._query_cts_batch(domains, order=order)


def _query_domainrel(
    domain: str,
    timeout: int = 10,
):
    """Convenience function to query a domain with DomainRel binary label."""
    client = CrediGraphClient(timeout=timeout)
    return client._query_domainrel(domain)


def _query_domainrel_batch(
    domains: list[str],
    order: str = "original",
    timeout: int = 10,
):
    """Convenience function to query domains with DomainRel binary labels."""
    client = CrediGraphClient(timeout=timeout)
    return client._query_domainrel_batch(domains, order=order)


def _query_dqr(
    domain: str,
    timeout: int = 10,
):
    """Convenience function to query a domain with GT DQR regression score."""
    client = CrediGraphClient(timeout=timeout)
    return client._query_dqr(domain)


def _query_dqr_batch(
    domains: list[str],
    order: str = "original",
    timeout: int = 10,
):
    """Convenience function to query domains with GT DQR regression score."""
    client = CrediGraphClient(timeout=timeout)
    return client._query_dqr_batch(domains, order=order)


