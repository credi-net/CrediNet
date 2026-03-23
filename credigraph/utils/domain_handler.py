import re
from typing import Dict, List, Optional, Iterable
from urllib.parse import urlparse
import tldextract

_extract = tldextract.TLDExtract(include_psl_private_domains=True)

DOMAIN_REGEX = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,}$"
)

def flip_domain(domain: str) -> str:
    """
    Transform a canonical 'domain.suffix' into the flipped form
    'suffix.domain', e.g.:

        'apnews.com'        -> 'com.apnews'
        'theregister.co.uk' -> 'co.uk.theregister'

    Requires the same `_extract` function used by `flip_if_needed`
    (e.g. tldextract with the Public Suffix List).
    """
    if not domain:
        return domain

    domain = domain.strip('.').lower()
    if not domain:
        return domain

    ext = _extract(domain)
    if not ext.domain or not ext.suffix:
        # If it can't be parsed canonically, return as-is
        return domain

    return f"{ext.suffix}.{ext.domain}"


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
    
    host = flip_domain(host)

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