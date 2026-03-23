import pytest
from credigraph import query, query_batch


@pytest.mark.integration
def test_single_query_helper(api_timeout: int) -> None:
    result = query("reuters.com", timeout=api_timeout)

    assert isinstance(result, dict)
    assert "domain" in result
    assert isinstance(result["domain"], str)
    if "credible" in result:
        assert isinstance(result["credible"], bool)


@pytest.mark.integration
def test_batch_query_helper(api_timeout: int) -> None:
    domains = ["reuters.com", "cbc.ca", "cnn.com"]
    results = query_batch(domains, timeout=api_timeout)

    assert isinstance(results, list)
    assert len(results) == len(domains)
    for item in results:
        assert isinstance(item, dict)
        assert "domain" in item
        assert isinstance(item["domain"], str)
        if "credible" in item:
            assert isinstance(item["credible"], bool)
