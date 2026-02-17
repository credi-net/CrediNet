<div align="center">


# CrediNet

<img src="img/credinet.png" alt="CrediNet Logo" style="width: 100px; height: auto;" />


Client-side CrediNet Service: <br> using CrediPred scores in practice for fact-checking and online retrieval.

</div>

<br>


## CrediBench Client

Client-side CrediNet Service: using domain-level <br> credibility scores ([CrediPred](https://github.com/credi-net/CrediPred/blob/main/README.md))  in practice for fact-checking and web retrieval.

</div>

---
<br>



### Install

```bash
pip install credigraph
```

### Usage

```python
from credigraph import query

# Single domain
result = query("apnews.com")
print(result)

# OR Multiple domains
results = query(["apnews.com", "cnn.com", "reuters.com"])
for result in results:
    print(result)
```

<!-- Set token as environment variable:
```bash
export HF_TOKEN=hf_your_token_here
```

Or pass it directly:
```python
from credigraph import CrediGraphClient

client = CrediGraphClient(token="hf_your_token_here")
result = client.query("reuters.com")
print(result)
``` -->

**Input Formats:**

The client normalizes various URL/domain formats automatically:
```python
# All of these work:
query("example.com")
query("www.example.com")
query("https://example.com/article")
query("EXAMPLE.COM")  # Case insensitive
```

**Check Version:**

```python
import credigraph
print(credigraph.__version__)
```

