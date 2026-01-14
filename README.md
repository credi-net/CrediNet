<div align="center">


# CrediNet

<img src="img/credinet.png" alt="CrediNet Logo" style="width: 100px; height: auto;" />


Client-side CrediNet Service: using domain-level <br> credibility scores ([CrediPred](https://github.com/credi-net/CrediPred/blob/main/README.md))  in practice for fact-checking and web retrieval.

</div>

---
<br>


## CrediBench Client [currently: not live]

Python client for the CrediGraph API.

### Install

```bash
pip install credigraph
```

### Usage

```python
from credigraph import query
print(query("apnews.com"))
print(query(["apnews.com", "cnn.com"]))
```

With auth: `export HF_TOKEN=hf_...` 

Or, 
```python
from credigraph import CrediGraphClient
c = CrediGraphClient(token="hf_...")
c.query("reuters.com")
```

