"""Validation utilities for Pixal3D API server."""

import uuid
from urllib.parse import urlparse


def validate_uid_format(uid: str) -> tuple[bool, str]:
    """
    Validate that ``uid`` matches an accepted task identifier format.

    The validation is intentionally strict to avoid path traversal and malformed
    identifiers entering Redis keys or file-system paths.

    Args:
        uid: The UID to validate.

    Returns:
        tuple[bool, str]: ``(is_valid, error_message)``.
    """
    if not uid:
        return False, "UID cannot be empty"

    # Check length first to reject obviously malformed values.
    if len(uid) > 100:
        return False, "UID exceeds maximum length"

    # Accept canonical UUID strings and 32-char hex UUID forms.
    try:
        uuid.UUID(uid)
        return True, ""

    except ValueError:
        pass

    # Also accept plain 32-char hex strings without hyphens.
    if len(uid) == 32 and all(char in "0123456789abcdefABCDEF" for char in uid):
        return True, ""

    # Accept short 6-char hex identifiers for compact node/task IDs.
    if len(uid) == 6 and all(char in "0123456789abcdefABCDEF" for char in uid):
        return True, ""

    return False, f"Invalid UID format: '{uid}'. Must be a valid UUID or hex string"


def validate_gcs_url(gs_url: str) -> bool:
    """
    Return True when string matches canonical gs://bucket/path format.

    The runtime accepts only canonical object-storage URLs here so queue
    payloads stay small and source resolution is deterministic.
    """
    if not isinstance(gs_url, str) or not gs_url.startswith("gs://"):
        return False

    parsed = urlparse(gs_url)
    return bool(parsed.netloc and parsed.path and parsed.path.strip("/"))


def validate_http_url(url: str) -> bool:
    """
    Return True when string is a valid HTTP/HTTPS URL.

    We keep this check intentionally narrow because remote fetches should only
    follow explicit web URLs, not arbitrary schemes.
    """
    if not isinstance(url, str):
        return False

    parsed = urlparse(url)
    return bool(parsed.scheme in ("http", "https") and parsed.netloc)
