import pytest
import requests
from credigraph.utils.domain_handler import (
    canonicalize_domain,
    flip_domain,
    normalize_domain,
    normalize_domains,
    normalize_domain_variants,
    unflip_domain,
)
from credigraph.client import CrediGraphClient


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Example.COM", "example.com"),
        (" https://www.Example.com/path?x=1 ", "example.com"),
        ("www.sub.example.co.uk", "sub.example.co.uk"),
        ("example.com/", "example.com"),
        ("example.com:443", "example.com"),
    ],
)
def test_normalize_domain_valid(raw, expected):
    assert normalize_domain(raw) == expected


def test_canonicalize_domain_valid():
    assert canonicalize_domain(" https://www.ApNews.com/article ") == "apnews.com"

def test_normalize_domains_deduplicates():
    raw = ["Example.com", "www.example.com", "example.com"]

    assert normalize_domains(raw) == ["example.com"]


def test_normalize_domain_variants_contains_canonical_and_flipped():
    assert normalize_domain_variants("apnews.com") == ["apnews.com", "com.apnews"]


def test_unflip_domain_simple():
    assert unflip_domain("com.apnews") == "apnews.com"


def test_unflip_domain_multilabel_suffix():
    assert unflip_domain("co.uk.theregister") == "theregister.co.uk"


def test_normalize_domain_variants_accepts_flipped_input():
    assert normalize_domain_variants("com.apnews") == ["com.apnews", "apnews.com"]

def test_flip_domain_simple():
    assert flip_domain("apnews.com") == "com.apnews"

def test_flip_domain_multilabel_suffix():
    assert flip_domain("theregister.co.uk") == "co.uk.theregister"

def test_flip_domain_invalid():
    assert flip_domain("localhost") == "localhost"


def test_client_query_domain_falls_back_to_flipped_form(monkeypatch):
    client = CrediGraphClient(api_url="https://example.test")
    calls = []

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                response = requests.Response()
                response.status_code = self.status_code
                raise requests.exceptions.HTTPError(response=response)

        def json(self):
            return self._payload

    def fake_get(url, headers, timeout):
        calls.append(url)
        if url.endswith("/by_domain/apnews.com"):
            return FakeResponse(404, {"detail": "not found"})
        return FakeResponse(
            200,
            {
                "domain": "com.apnews",
                "pc1_score": 0.4552963078,
                "binary_score": 0,
            },
        )

    monkeypatch.setattr(requests, "get", fake_get)

    result = client.query_domain("https://www.apnews.com")

    assert calls == [
        "https://example.test/by_domain/apnews.com",
        "https://example.test/by_domain/com.apnews",
    ]
    assert result["domain"] == "com.apnews"
    assert result["continuous_score"] == 0.46
    assert result["binary_score"] == 0.0


def test_client_label_sets(monkeypatch):
    client = CrediGraphClient(api_url="https://example.test")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "label_sets": [
                    {
                        "name": "DomainRel",
                        "task": "regression",
                        "label_field": "pc1",
                        "source_url": "https://huggingface.co/datasets/credi-net/DomainRel",
                        "splits": [{"name": "train", "path": "regression/train_regression_domains.parquet"}],
                    }
                ]
            }

    def fake_get(url, headers, timeout):
        assert url == "https://example.test/label_sets"
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    result = client.label_sets()

    assert result["label_sets"][0]["name"] == "DomainRel"
    assert result["label_sets"][0]["label_field"] == "pc1"


def test_client_help(monkeypatch):
    client = CrediGraphClient(api_url="https://example.test")

    class FakeResponse:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, headers, timeout):
        if url.endswith("/metadata"):
            return FakeResponse(
                {
                    "api_version": "0.2.0",
                    "data_cutoff_month": "2024-12",
                    "method": "GAT-TEXT",
                    "score_sources": {"regression": "x", "binary": "y"},
                }
            )
        if url.endswith("/openapi.json"):
            return FakeResponse(
                {
                    "paths": {
                        "/by_domain/{domain}": {
                            "get": {
                                "summary": "Query credibility scores for a single domain",
                                "description": "Returns credibility metrics for one or multiple domains.",
                            }
                        }
                    }
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)

    result = client.help()

    assert result["source"] == "openapi"
    assert result["api_version"] == "0.2.0"
    assert result["endpoints"][0]["path"] == "/by_domain/{domain}"
    assert result["endpoints"][0]["method"] == "GET"