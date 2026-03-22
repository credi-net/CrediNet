import pytest
from credigraph import CrediGraphClient


@pytest.fixture(scope="session")
def api_timeout() -> int:
    return 30


@pytest.fixture(scope="session")
def client(api_timeout: int) -> CrediGraphClient:
    return CrediGraphClient(timeout=api_timeout)


@pytest.fixture(scope="session")
def sample_single_domain() -> str:
    return "apnews.com"


@pytest.fixture(scope="session")
def sample_batch_domains() -> list[str]:
    return ["apnews.com", "cnn.com", "reuters.com"]
