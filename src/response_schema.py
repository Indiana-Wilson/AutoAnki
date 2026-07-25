"""Shared constraints for OpenAI structured-response schema identifiers."""

import hashlib


MAX_RESPONSE_FORMAT_NAME_LENGTH = 64


def bounded_response_format_name(name):
    """Keep a stable readable prefix while satisfying the API's 64-char cap."""
    if not isinstance(name, str) or not name:
        raise ValueError("A response-format name must be nonempty text.")
    if len(name) <= MAX_RESPONSE_FORMAT_NAME_LENGTH:
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
    prefix_length = MAX_RESPONSE_FORMAT_NAME_LENGTH - len(digest) - 1
    return f"{name[:prefix_length]}_{digest}"
