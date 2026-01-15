import pytest
from credigraph.utils.string_handler import normalize_domain, normalize_domains
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

@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not a domain",
        "ftp://example.com",
        "http://256.256.256.256",
    ],
)
def test_normalize_domain_invalid(raw):
    with pytest.raises(ValueError):
        normalize_domain(raw)

def test_normalize_domains_deduplicates():
    raw = ["Example.com", "www.example.com", "example.com"]
    assert normalize_domains(raw) == ["example.com"]

def test_query_domain(requests_mock):
    requests_mock.get(
        "https://credi-net-credinet.hf.space/by_domain/example.com",
        json={"domain": "example.com", "score": 0.42},
        status_code=200,
    )

    client = CrediGraphClient()
    result = client.query_domain("https://www.Example.com")

    assert result["domain"] == "example.com"