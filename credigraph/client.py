import os
import requests
from typing import Any, Dict, List
from credigraph.utils import normalize_domain_variants, normalize_domains

DEFAULT_API_URL = "https://credi-net-credinet.hf.space"
DEFAULT_METHOD = "GAT-TEXT"
DEFAULT_DATA_CUTOFF_MONTH = "2024-12"
DEFAULT_REGRESSION_SOURCE = (
    "https://huggingface.co/datasets/credi-net/CrediPred/blob/main/MLP-Inference/"
    "Text%2BGAT/mlpInfer_reg_dec2024_embeddinggemma-300m_GAT-text_scores.parquet.parquet"
)
DEFAULT_BINARY_SOURCE = (
    "https://huggingface.co/datasets/credi-net/CrediPred/blob/main/MLP-Inference/"
    "Text%2BGAT/mlpInfer_binaryClassifcation_dec2024_embeddinggemma-300m_GAT-text_scores.parquet"
)
DEFAULT_LABEL_SETS = {
    "label_sets": [
        {
            "name": "DomainRel",
            "task": "classification",
            "label_field": "binary",
            "description": "Binary label set of web domains' credibility.",
            "source_url": "https://huggingface.co/datasets/credi-net/DomainRel",
            "format": "parquet",
            "domain_count": None,
        }
    ]
}

DEFAULT_ENDPOINT_HELP = [
    {
        "path": "/",
        "method": "GET",
        "summary": "Root status and docs location",
    },
    {
        "path": "/health",
        "method": "GET",
        "summary": "Health check endpoint",
    },
    {
        "path": "/metadata",
        "method": "GET",
        "summary": "API metadata and model provenance",
    },
    {
        "path": "/by_domain/{domain}",
        "method": "GET",
        "summary": "Query credibility scores for a single domain",
    },
    {
        "path": "/stats",
        "method": "GET",
        "summary": "Monthly graph composition statistics",
    },
    {
        "path": "/months",
        "method": "GET",
        "summary": "Available monthly datasets and download locations",
    },
    {
        "path": "/label_sets",
        "method": "GET",
        "summary": "Available label-set metadata",
    },
]


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

    def _get_json(self, path: str) -> Dict[str, Any]:
        url = f"{self.api_url}{path}"
        headers = self._get_headers()

        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(
                f"Request to {url} timed out after {self.timeout}s"
            )

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
        variants = normalize_domain_variants(domain)
        last_error: requests.exceptions.HTTPError | None = None

        for candidate in variants:
            url = f"{self.api_url}/by_domain/{candidate}"
            headers = self._get_headers()

            try:
                r = requests.get(url, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                result = r.json()

                if "pc1_score" in result and "continuous_score" not in result:
                    result["continuous_score"] = result.pop("pc1_score")
                
                if "continuous_score" in result and isinstance(result["continuous_score"], (int, float)):
                    result["continuous_score"] = float(f"{result['continuous_score']:.2f}")

                if "binary_score" not in result:
                    legacy_binary_keys = (
                        "binary_classification_score",
                        "binaryScore",
                        "binary",
                    )
                    for key in legacy_binary_keys:
                        if key in result:
                            result["binary_score"] = result.pop(key)
                            break

                if "binary_score" in result and isinstance(result["binary_score"], bool):
                    result["binary_score"] = int(result["binary_score"])

                if "binary_score" in result and isinstance(result["binary_score"], (int, float)):
                    result["binary_score"] = float(f"{result['binary_score']:.2f}")
                
                return result
            except requests.exceptions.Timeout:
                raise requests.exceptions.Timeout(
                    f"Request to {url} timed out after {self.timeout}s"
                )
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    last_error = exc
                    continue
                raise

        if last_error is not None:
            raise last_error

        raise RuntimeError("No domain variants were generated for lookup")

    def metadata(self) -> Dict[str, Any]:
        """Return API metadata including version, data cutoff, method, and score sources."""
        result = self._get_json("/metadata")

        result.setdefault("method", DEFAULT_METHOD)
        result.setdefault("data_cutoff_month", DEFAULT_DATA_CUTOFF_MONTH)
        result.setdefault(
            "score_sources",
            {
                "regression": DEFAULT_REGRESSION_SOURCE,
                "binary": DEFAULT_BINARY_SOURCE,
            },
        )
        return result

    def health(self) -> Dict[str, Any]:
        """Return API health status and server version."""
        return self._get_json("/health")

    def stats(self) -> Dict[str, Any]:
        """Return monthly graph composition stats for all available datasets."""
        return self._get_json("/stats")

    def months(self) -> Dict[str, Any]:
        """Return downloadable monthly dataset entries and metadata."""
        return self._get_json("/months")

    def label_sets(self) -> Dict[str, Any]:
        """Return metadata for the currently exposed label sets."""
        try:
            return self._get_json("/label_sets")
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return DEFAULT_LABEL_SETS
            raise

    def list_months(self) -> Dict[str, Any]:
        """Alias for months()."""
        return self.months()

    def dataset_stats(self) -> Dict[str, Any]:
        """Alias for stats()."""
        return self.stats()

    def summarize(self) -> Dict[str, Any]:
        """Alias for metadata()."""
        return self.metadata()

    def labels(self) -> Dict[str, Any]:
        """Alias for label_sets()."""
        return self.label_sets()

    def help(self) -> Dict[str, Any]:
        """Return a compact endpoint guide for the configured API."""
        payload: Dict[str, Any] = {
            "api_url": self.api_url,
            "client_user_agent": self._user_agent,
        }

        try:
            meta = self.metadata()
            payload["api_version"] = meta.get("api_version")
            payload["method"] = meta.get("method")
            payload["data_cutoff_month"] = meta.get("data_cutoff_month")
        except requests.exceptions.RequestException:
            pass

        try:
            spec = self._get_json("/openapi.json")
            endpoints: List[Dict[str, Any]] = []
            for path, operations in spec.get("paths", {}).items():
                for verb in ("get", "post", "put", "patch", "delete"):
                    op = operations.get(verb)
                    if not op:
                        continue
                    endpoints.append(
                        {
                            "path": path,
                            "method": verb.upper(),
                            "summary": op.get("summary"),
                            "description": op.get("description"),
                        }
                    )

            if endpoints:
                payload["endpoints"] = endpoints
                payload["source"] = "openapi"
                return payload
        except requests.exceptions.RequestException:
            pass

        payload["endpoints"] = DEFAULT_ENDPOINT_HELP
        payload["source"] = "fallback"
        return payload

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
