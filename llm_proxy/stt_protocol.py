"""Shared STT request-header protocol helpers."""

import base64
import binascii


INITIAL_PROMPT_HEADER = "X-Initial-Prompt"
ENCODED_INITIAL_PROMPT_HEADER = "X-Initial-Prompt-Encoded"
_INITIAL_PROMPT_ENCODING_PREFIX = "v1:"
INITIAL_PROMPT_HEADERS = (
    INITIAL_PROMPT_HEADER,
    ENCODED_INITIAL_PROMPT_HEADER,
)


class InitialPromptEncodingError(ValueError):
    """An encoded STT initial prompt is malformed or ambiguous."""


def decode_initial_prompt(headers):
    plain = headers.get(INITIAL_PROMPT_HEADER)
    encoded = headers.get(ENCODED_INITIAL_PROMPT_HEADER)
    if plain is not None and encoded is not None:
        raise InitialPromptEncodingError(
            "both plain and encoded initial prompt headers are present"
        )
    if encoded is None:
        return plain
    if not encoded.startswith(_INITIAL_PROMPT_ENCODING_PREFIX):
        raise InitialPromptEncodingError("unsupported initial prompt encoding")
    payload = encoded[len(_INITIAL_PROMPT_ENCODING_PREFIX):]
    try:
        prompt_bytes = base64.b64decode(payload, validate=True)
        return prompt_bytes.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise InitialPromptEncodingError("malformed encoded initial prompt") from exc


def forward_initial_prompt_headers(source, target):
    """Copy either prompt representation without interpreting it."""
    for name in INITIAL_PROMPT_HEADERS:
        value = source.get(name)
        if value is not None:
            target[name] = value
