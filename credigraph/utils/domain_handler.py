import re
from typing import List, Iterable
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


def unflip_domain(domain: str) -> str:
    """Transform a flipped domain like 'com.apnews' back to 'apnews.com' when possible."""
    if not domain:
        return domain

    labels = domain.strip('.').lower().split('.')
    if len(labels) < 2:
        return domain

    for split_index in range(len(labels) - 1, 0, -1):
        suffix = ".".join(labels[:split_index])
        candidate = ".".join(labels[split_index:] + labels[:split_index])
        extracted = _extract(candidate)
        if extracted.suffix == suffix:
            return candidate

    return domain


def canonicalize_domain(value: str) -> str:
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


def normalize_domain(value: str) -> str:
    """Normalize a domain input into the canonical host form."""
    return canonicalize_domain(value)


def normalize_domain_variants(value: str) -> List[str]:
    """Return canonical and flipped variants for APIs that may use either convention."""
    normalized = canonicalize_domain(value)
    canonical = unflip_domain(normalized)
    flipped = flip_domain(canonical)

    variants = []
    for candidate in (normalized, canonical, flipped):
        if candidate and candidate not in variants:
            variants.append(candidate)

    return variants

def normalize_domains(domains: Iterable[str]) -> List[str]:
    seen = set()
    clean = []
    for d in domains:
        nd = normalize_domain(d)
        if nd not in seen:
            seen.add(nd)
            clean.append(nd)
    return clean