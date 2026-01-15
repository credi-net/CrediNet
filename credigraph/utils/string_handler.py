import os
import re
import requests
from typing import List, Iterable
from urllib.parse import urlparse

DOMAIN_REGEX = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$"
)

def normalize_domain(value: str) -> str:
    if not value or not isinstance(value, str):
        raise ValueError("Domain must be a non-empty string")

    value = value.strip().lower()

    if not value.startswith(("http://", "https://")):
        value = "http://" + value

    parsed = urlparse(value)

    host = parsed.hostname or ""
    if host.startswith("www."):
        host = host[4:]

    if not DOMAIN_REGEX.match(host):
        raise ValueError(f"Invalid domain: {host}")

    return host

def normalize_domains(domains: Iterable[str]) -> List[str]:
    seen = set()
    clean = []
    for d in domains:
        nd = normalize_domain(d)
        if nd not in seen:
            seen.add(nd)
            clean.append(nd)
    return clean