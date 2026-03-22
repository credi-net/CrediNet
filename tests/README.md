# Test Suite

- **`test_api.py`** - Main test suite validating API client behavior against OpenAPI spec
- **`test_helper.py`** - Pytest unit / integrations tests for query functions 
- **`conftest.py`** - Shared pytest fixtures (client, timeout, sample domains)


### Run all tests
```bash
test_all.sh
```

Or run individual test files:
```bash
pytest -q test_api.py test_helper.py
```
