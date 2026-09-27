"""Tests for Stage A VLM clients, cache helpers, and Ollama wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from lsi_vlm.clients import (
    CacheKey,
    OllamaClient,
    VLMClient,
    append_cache_record,
    cache_path_for,
    hash_tile_payload,
    load_cache_index,
    lookup_cache,
    make_client,
    prompt_hash,
)
from lsi_vlm.prompts import parse_vlm_response


class _FakeClient(VLMClient):
    backend = "fake"

    def __init__(self, *, replies: list[str] | None = None, fail_times: int = 0, **kwargs):
        super().__init__(model=kwargs.pop("model", "fake-model"), **kwargs)
        self.replies = list(replies or [])
        self.fail_times = fail_times
        self.calls = 0

    def _complete_once(self, images: Sequence[bytes | str | Path], prompt: str) -> str:
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("transient failure")
        if not self.replies:
            return json.dumps(
                {
                    "class": "moderate",
                    "score": 50,
                    "key_factors": ["slope"],
                    "reason": "Default fake reply.",
                }
            )
        return self.replies.pop(0)


def test_cache_key_and_jsonl_roundtrip(tmp_path: Path):
    key = CacheKey(tile_hash="abc", prompt_hash="def", model="qwen2.5vl")
    path = cache_path_for(key.model, tmp_path)
    record = {
        **key.as_dict(),
        "key_id": key.key_id,
        "backend": "ollama",
        "raw_text": '{"class":"low","score":20,"key_factors":[],"reason":"ok"}',
        "prediction": {
            "class": "low",
            "score": 20,
            "key_factors": [],
            "reason": "ok",
        },
        "api_key": "SHOULD_NOT_PERSIST",
    }
    append_cache_record(path, record)
    index = load_cache_index(path)
    assert key.key_id in index
    assert "api_key" not in index[key.key_id]
    assert lookup_cache(tmp_path, key) is not None


def test_complete_uses_cache_on_second_call(tmp_path: Path):
    good = json.dumps(
        {
            "class": "high",
            "score": 80,
            "key_factors": ["built-up"],
            "reason": "Urban cut slope.",
        }
    )
    client = _FakeClient(
        replies=[good],
        cache_dir=tmp_path,
        use_cache=True,
        max_retries=1,
        model="unit-test-model",
    )
    img = b"\x89PNG\r\n\x1a\nfake"
    r1 = client.complete([img], "prompt-a", tile_hash="tile1")
    assert r1.cached is False
    assert r1.prediction is not None
    assert r1.prediction.class_ == "high"
    assert client.calls == 1

    r2 = client.complete([img], "prompt-a", tile_hash="tile1")
    assert r2.cached is True
    assert r2.prediction is not None
    assert r2.prediction.score == 80
    assert client.calls == 1  # no second live call


def test_retry_then_success(tmp_path: Path):
    client = _FakeClient(
        fail_times=2,
        cache_dir=tmp_path,
        use_cache=False,
        max_retries=3,
        retry_backoff_s=0.01,
        model="retry-model",
    )
    result = client.complete([b"img"], "p", tile_hash="t")
    assert result.prediction is not None
    assert client.calls == 3
    assert result.error is None


def test_parse_failure_is_logged_not_silent(tmp_path: Path, caplog):
    import logging

    client = _FakeClient(
        replies=["not json at all"],
        cache_dir=tmp_path,
        use_cache=False,
        max_retries=1,
        model="bad-parse",
    )
    with caplog.at_level(logging.ERROR):
        result = client.complete([b"x"], "p", tile_hash="t")
    assert result.prediction is None
    assert result.raw_text == "not json at all"
    assert any("parse" in r.message.lower() for r in caplog.records)


def test_make_client_ollama_defaults():
    client = make_client("ollama", use_cache=False, cache_dir="outputs_vlm/cache")
    assert isinstance(client, OllamaClient)
    assert client.model == "qwen2.5vl"
    assert client.temperature == 0.0


def test_make_client_unknown_backend():
    with pytest.raises(ValueError, match="Unknown VLM backend"):
        make_client("watson")


def test_hash_helpers_stable():
    a = hash_tile_payload(b"abc", "meta")
    b = hash_tile_payload(b"abc", "meta")
    c = hash_tile_payload(b"abc", "other")
    assert a == b
    assert a != c
    assert prompt_hash("hello") == prompt_hash("hello")
    assert prompt_hash("hello") != prompt_hash("hello!")


def test_ollama_complete_once_posts_expected_payload(tmp_path: Path, monkeypatch):
    """Stub httpx so we never hit a real Ollama daemon (httpx may be optional)."""
    import sys
    import types

    class _FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "class": "moderate",
                            "score": 55,
                            "key_factors": ["slope"],
                            "reason": "Stubbed Ollama reply.",
                        }
                    )
                }
            }

    posted: dict = {}

    class _FakeHttpClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json):  # noqa: A002
            posted["url"] = url
            posted["payload"] = json
            assert url.endswith("/api/chat")
            assert json["model"] == "qwen2.5vl"
            assert json["options"]["temperature"] == 0.0
            assert json["messages"][0]["images"]
            return _FakeResp()

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Client = _FakeHttpClient
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    client = OllamaClient(
        model="qwen2.5vl",
        cache_dir=tmp_path,
        use_cache=True,
        max_retries=1,
    )
    result = client.complete([b"\x89PNG"], "Assess this tile.", tile_hash="tile-ollama")
    assert result.prediction is not None
    assert result.prediction.class_ == "moderate"
    assert result.cached is False
    assert posted["url"].endswith("/api/chat")
    # Second call hits cache
    result2 = client.complete([b"\x89PNG"], "Assess this tile.", tile_hash="tile-ollama")
    assert result2.cached is True


def test_parse_vlm_response_exported_for_clients():
    pred = parse_vlm_response(
        '{"class":"low","score":10,"key_factors":["a"],"reason":"Short."}'
    )
    assert pred is not None
