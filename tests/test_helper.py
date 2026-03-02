#!/usr/bin/env python3
"""
Helper script to test API queries with sample domains.
"""

from credigraph import CrediGraphClient
import json


def test_single_query():
    """Query single domain: reuters.com"""
    print("\nSingle domain query: reuters.com")
    
    client = CrediGraphClient()
    result = client.query("reuters.com")
    
    print(json.dumps(result, indent=2))
    print()


def test_batch_query():
    """Query multiple domains: reuters.com, cbc.ca, cnn.com"""
    print("Batch query: reuters.com, cbc.ca, cnn.com")
    
    client = CrediGraphClient()
    domains = ["reuters.com", "cbc.ca", "cnn.com"]
    results = client.query(domains)
    
    for result in results:
        print(json.dumps(result, indent=2))
        print()


if __name__ == "__main__":
    test_single_query()
    test_batch_query()
