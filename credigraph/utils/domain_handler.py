import re
from typing import Dict, List, Optional
from urllib.parse import urlparse
import pandas as pd 
import tldextract

_extract = tldextract.TLDExtract(include_psl_private_domains=True)
_DOMAIN_CLEAN_RE = re.compile(r"^https?://", re.IGNORECASE)

def normalize_domain(d: str | None) -> str | None:
    """Normalize a domain string.

    Parameters:
        d : str or None
            Domain string.

    Returns:
        str or None
            Normalized domain string without "www." prefix.
    """
    if d is None or pd.isna(d):
        return None
    d = d.lower().strip()
    d = _DOMAIN_CLEAN_RE.sub("", d)
    d = d.split("/", 1)[0]
    d = d.split("?", 1)[0]
    d = d.split("#", 1)[0]
    d = d.split(":", 1)[0]

    if d.startswith("www."):
        d = d[4:]
    return d 


def flip_if_needed(domain: str) -> str:
    """Normalize a possibly flipped domain (e.g., 'co.uk.theregister') into the
    canonical 'domain.suffix' (e.g., 'theregister.co.uk').

    Parameters:
        domain : str
            Input domain string.

    Returns:
        str
            Normalized domain in "domain.suffix" form.
    """
    if pd.isna(domain):
        return domain

    domain = str(domain).strip('.').lower()
    if not domain:
        return domain

    labels = [p for p in domain.split('.') if p]
    Strategy: try all cyclic rotations of labels; pick the parse with
    the longest PSL suffix (# of labels), then longest domain label.
    """
    if not domain:
        return domain

    labels = [p for p in domain.strip('.').lower().split('.') if p]
    if not labels:
        return domain

    best = None  # (suffix_label_count, domain_len, normalized_str)

    n = len(labels)
    for r in range(n):
        # rotation: move the last r labels to the front
        rotated = labels[-r:] + labels[:-r] if r else labels
        rotated_str = '.'.join(rotated)

        ext = _extract(rotated_str)
        if not ext.suffix or not ext.domain:
            continue

        suffix_labels = ext.suffix.count('.') + 1  # e.g., 'co.uk' -> 2
        dom_len = len(ext.domain)
        normalized = f'{ext.domain}.{ext.suffix}'

        cand = (suffix_labels, dom_len, normalized)
        if (best is None) or (cand > best):
            best = cand

    # Fall back: parse the input as-is if no rotation produced a valid suffix
    if best is None:
        ext = _extract('.'.join(labels))
        if ext.suffix and ext.domain:
            return f'{ext.domain}.{ext.suffix}'
        return '.'.join(labels)

    return best[2]


def lookup(domain: str, dqr_domains: Dict[str, List[float]]) -> Optional[List[float]]:
    """Look up a domain by exact canonical match (with normalization).

    Parameters:
        domain : str
            Domain to search for.
        dqr_domains : dict[str, list[float]]
            Mapping from known domain strings to associated metric lists.

    Returns:
        list[float] or None
            Associated metric list if found, otherwise None.
    """
    domain_name = flip_if_needed(domain)
    return dqr_domains.get(domain_name)


def reverse_domain(domain: str) -> str:
    """Reverse the label order of a domain string.

    Parameters:
        domain : str
            Domain string.

    Returns:
        str
            Domain with label order reversed.
    """
    return '.'.join(domain.split('.')[::-1])


def extract_domain(raw: str) -> str | None:
    """Extract and normalize a domain from a raw string or URL.

    The input may be a bare domain, a URL, or a malformed string. The function
    attempts to normalize the input and extract a valid domain if possible.

    Parameters:
        raw : str
            Raw input string.

    Returns:
        str or None
            Extracted domain string, or None if extraction fails.
    """
    if not raw:
        return None

    raw = raw.strip().strip('\'"')
    raw = raw.replace('&amp;', '&')

    # if no scheme, add one so urlparse works
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', raw):
        raw = 'http://' + raw

    try:
        parsed = urlparse(raw)
        domain = parsed.netloc.lower()

        if not domain:
            return None

        domain = domain.split(':', 1)[0]

        if any(c.isspace() for c in domain):
            return None

        if '.' not in domain:
            return None

        return domain

    except Exception:
        return None
        
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

def lookup(domain: str, dqr_domains: Dict[str, List[float]]) -> Optional[List[float]]:
    """Look up domain in dqr_domains, return associated data if found."""
    domain_parts = domain.split('.')
    for key, value in dqr_domains.items():
        key_parts = key.split('.')
        if (
            len(key_parts) >= 2
            and key_parts[0] in domain_parts
            and key_parts[1] in domain_parts
        ):
            return value
    return None


def lookup_exact(
    domain: str, dqr_domains: Dict[str, List[float]]
) -> Optional[List[float]]:
    domain_name = flip_if_needed(domain)
    return dqr_domains.get(domain_name)


def reverse_domain(domain: str) -> str:
    return '.'.join(domain.split('.')[::-1])
