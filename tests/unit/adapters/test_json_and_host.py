import pytest

from adapters.llm.host import normalize_host
from adapters.llm.json_parsing import extract_json


@pytest.mark.parametrize("text, expected", [
    ('{"a": 1}', {"a": 1}),
    ('Sure! ```json\n{"a": 2}\n``` hope that helps', {"a": 2}),
    ('The answer is {"a": {"b": 3}} done', {"a": {"b": 3}}),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_extract_json_without_json_raises():
    with pytest.raises(ValueError):
        extract_json("no json here")


@pytest.mark.parametrize("host, url", [
    ("0.0.0.0", "http://localhost:11434"),
    ("0.0.0.0:9999", "http://localhost:9999"),
    ("localhost", "http://localhost:11434"),
    ("http://gpu-box:11434/", "http://gpu-box:11434"),
    ("https://ollama.internal", "https://ollama.internal:11434"),
    ("", "http://localhost:11434"),
])
def test_normalize_host(host, url):
    assert normalize_host(host) == url
