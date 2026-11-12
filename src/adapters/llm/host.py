"""Ollama host normalisation."""

from __future__ import annotations

DEFAULT_PORT = "11434"


def normalize_host(host: str) -> str:
    """
    Turn an OLLAMA_HOST value into a client URL. The variable is often set to
    a server bind address such as ``0.0.0.0`` or ``0.0.0.0:11434``, which is
    not connectable; map it to localhost and fill in scheme and port.
    """
    host = host.strip().rstrip("/") or "localhost"
    scheme = "http"
    if "://" in host:
        scheme, host = host.split("://", 1)
    hostname, _, port = host.partition(":")
    if hostname in ("", "0.0.0.0", "[::]", "::"):
        hostname = "localhost"
    return f"{scheme}://{hostname}:{port or DEFAULT_PORT}"
