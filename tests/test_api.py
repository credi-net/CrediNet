#!/usr/bin/env python3
"""
Test the Python client against the real API.
Validates that responses match the OpenAPI spec promises.
"""

from credigraph import CrediGraphClient
import sys


def test_single_domain():
    """Test: Query single domain"""
    print("\n[1/5] Testing single domain query")
    
    client = CrediGraphClient()
    result = client.query_domain("apnews.com")
    
    # Check required fields exist
    assert "domain" in result, "Missing 'domain' field"
    assert "continuous_score" in result, "Missing 'continuous_score' field (should be renamed from pc1_score)"
    assert "binary_score" in result, "Missing 'binary_score' field"
    print(f"   Got domain: {result['domain']}")
    print(f"   Got continuous score: {result['continuous_score']}")
    print(f"   Got binary score: {result['binary_score']}")
    
    # Check types match spec
    assert isinstance(result["domain"], str), f"domain should be string, got {type(result['domain'])}"
    assert isinstance(result["continuous_score"], (int, float)), \
        f"continuous_score should be number, got {type(result['continuous_score'])}"
    assert isinstance(result["binary_score"], (int, float)), \
        f"binary_score should be number, got {type(result['binary_score'])}"
    print("   Data types correct")
    
    # Check score is in valid range (0-1)
    score = result["continuous_score"]
    assert 0 <= score <= 1, f"continuous_score should be 0-1, got {score}"
    binary = result["binary_score"]
    assert 0 <= binary <= 1, f"binary_score should be 0-1, got {binary}"
    print("   Score in valid range [0, 1]")
    
    # Check score has max 2 decimal places
    assert round(score, 2) == score, f"continuous_score should have max 2 decimal places, got {score}"
    assert round(binary, 2) == binary, f"binary_score should have max 2 decimal places, got {binary}"
    print("   Score formatted to max 2 decimal places")


def test_multiple_domains():
    """Test: Query multiple domains"""
    print("\n[2/5] Testing multiple domains query")
    
    client = CrediGraphClient()
    domains = ["apnews.com", "cnn.com", "reuters.com"]
    results = client.query(domains)
    
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) == len(domains), f"Expected {len(domains)} results, got {len(results)}"
    print(f"   Got {len(results)} results")
    
    # Check each result
    for i, result in enumerate(results):
        assert "domain" in result, f"Result {i}: Missing 'domain'"
        assert "continuous_score" in result, f"Result {i}: Missing 'continuous_score'"
        assert "binary_score" in result, f"Result {i}: Missing 'binary_score'"
        assert isinstance(result["continuous_score"], (int, float)), \
            f"Result {i}: continuous_score not a number"
        assert isinstance(result["binary_score"], (int, float)), \
            f"Result {i}: binary_score not a number"
        assert 0 <= result["continuous_score"] <= 1, \
            f"Result {i}: continuous_score not in range [0, 1]"
        assert 0 <= result["binary_score"] <= 1, \
            f"Result {i}: binary_score not in range [0, 1]"
        assert round(result["continuous_score"], 2) == result["continuous_score"], \
            f"Result {i}: continuous_score should have max 2 decimal places, got {result['continuous_score']}"
        assert round(result["binary_score"], 2) == result["binary_score"], \
            f"Result {i}: binary_score should have max 2 decimal places, got {result['binary_score']}"
    
    print("   All results have required fields")
    print("   All scores in valid range")
    print("   All scores formatted to max 2 decimal places")


def test_user_agent_header():
    """Test: Client sends proper User-Agent header"""
    print("\n[3/5] Testing User-Agent versioning header")
    
    client = CrediGraphClient()
    headers = client._get_headers()
    
    assert "User-Agent" in headers, "Missing User-Agent header"
    user_agent = headers["User-Agent"]
    
    assert user_agent.startswith("credigraph/"), f"User-Agent should start with 'credigraph/', got '{user_agent}'"
    print(f"   User-Agent: {user_agent}")
    
    # Check version format (semantic versioning X.Y.Z)
    version = user_agent.split("/")[1]
    parts = version.split(".")
    assert len(parts) >= 2, f"Version should be X.Y.Z format, got {version}"
    print(f"   Version format valid: {version}")


def test_client_configuration():
    """Test: supported client parameter configuration"""
    print("\n[4/5] Testing client configuration")

    client_timeout = CrediGraphClient(timeout=30)
    assert client_timeout.timeout == 30, "timeout not set correctly"
    print("   Can specify custom timeout")


def test_metadata_endpoint():
    """Test: Metadata endpoint includes required model provenance fields"""
    print("\n[5/5] Testing metadata endpoint")

    client = CrediGraphClient()
    result = client.metadata()

    assert "api_version" in result, "Missing 'api_version'"
    assert "data_cutoff_month" in result, "Missing 'data_cutoff_month'"
    assert "method" in result, "Missing 'method'"
    assert "score_sources" in result, "Missing 'score_sources'"
    assert "regression" in result["score_sources"], "Missing regression source URL"
    assert "binary" in result["score_sources"], "Missing binary source URL"

    print(f"   API version: {result['api_version']}")
    print(f"   Data cutoff: {result['data_cutoff_month']}")
    print(f"   Method: {result['method']}")


def main():
    """Run all tests"""
    print("\nCrediGraph Client Test Suite")
    print("Testing: Python client + Real API\n")
    
    try:
        test_single_domain()
        test_multiple_domains()
        test_user_agent_header()
        test_client_configuration()
        test_metadata_endpoint()
        
        print("\n[PASS] All client tests passed\n")
        return 0
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}\n")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
