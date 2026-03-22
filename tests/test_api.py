import pytest

@pytest.mark.integration
def test_single_domain(client, sample_single_domain: str) -> None:
    result = client.query(sample_single_domain)

    assert "domain" in result
    assert "credible" in result
    assert isinstance(result["domain"], str)
    assert isinstance(result["credible"], bool)


@pytest.mark.integration
def test_multiple_domains(client, sample_batch_domains: list[str]) -> None:
    results = client.query_batch(sample_batch_domains)

    assert isinstance(results, list)
    assert len(results) == len(sample_batch_domains)

    found_with_binary = 0
    for index, result in enumerate(results):
        assert "domain" in result, f"Result {index}: missing 'domain'"
        assert isinstance(result["domain"], str)
        if "credible" in result:
            assert isinstance(result["credible"], bool)
            found_with_binary += 1

    assert found_with_binary > 0, "Expected at least one batch result with a 'credible' field"


def test_user_agent_header(client) -> None:
    headers = client._get_headers()

    assert "User-Agent" in headers
    user_agent = headers["User-Agent"]
    assert user_agent.startswith("credigraph/")
