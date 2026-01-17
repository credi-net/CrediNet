import pytest
from credigraph.utils.string_handler import normalize_domain, normalize_domains
from credigraph.utils.domain_handler import flip_domain
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
    assert normalize_domain(raw) == flip_domain(expected)

def test_normalize_domains_deduplicates():
    raw = ["Example.com", "www.example.com", "example.com"]

    assert normalize_domains(raw) == [flip_domain("example.com")]

def test_flip_domain_simple():
    assert flip_domain("apnews.com") == "com.apnews"

def test_flip_domain_multilabel_suffix():
    assert flip_domain("theregister.co.uk") == "co.uk.theregister"

def test_flip_domain_invalid():
    assert flip_domain("localhost") == "localhost"