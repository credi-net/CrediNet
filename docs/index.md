# CrediGraph

Domain credibility scoring client for fact-checking and online retrieval.

[docs in progress]

 
## Overview

* CrediGraph API: query credibility scores from 45M+ domains based on our content- and topology-based methodology. 
* Datasets: find our available datasets on Huggingface.
* TAG Construction Pipeline: our optimized graph construction pipeline structures Common Crawl and web-scraped data into Text-Attributed temporal Graphs (TAGs). 

## Install

```bash
pip install credigraph
```

## Quick Start

```python
from credigraph import query

# Single domain
result = query("apnews.com")
print(result["continuous_score"])

# Multiple domains
results = query(["apnews.com", "cnn.com", "reuters.com"])
for result in results:
    print(result["domain"], result["continuous_score"])
```
