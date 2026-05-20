# API Configuration

OpenAPI specification, client configuration, and support information.

---

## Specification

**OpenAPI Specification:** [openapi.yaml](../openapi.yaml)

The REST API is fully documented in OpenAPI 3.0 format. Use this for:
- HTTP endpoint definitions
- Request/response schemas
- API status codes and error handling

---

## Client Configuration

### Timeout

All query functions accept an optional `timeout` parameter (in seconds):

```python
from credigraph import query, CrediGraphClient

# Function level
result = query("apnews.com", timeout=15)

# Client level
client = CrediGraphClient(timeout=15)
result = client.query("apnews.com")
```

Default: 10 seconds

---

## Health Checks

Check API health before querying:

```python
import requests
from credigraph import CrediGraphClient

client = CrediGraphClient()
response = requests.post(f"{client.api_url}/health")
# {"status": "ok", "api_version": "0.4.1"}
```

---

## Development

- **Testing Guide:** [credigraph/README.md](https://github.com/credi-net/CrediNet/blob/main/credigraph/README.md)
- **Main README:** [README.md](https://github.com/credi-net/CrediNet/blob/main/README.md)

---

## Support

- **Issues & Bug Reports:** [GitHub Issues](https://github.com/credi-net/CrediNet/issues)
- **Questions:** [Contact us](mailto:emma.kondrup@mila.quebec)
- **API Status:** [Hugging Face Spaces](https://huggingface.co/spaces/credi-net/CrediNet)