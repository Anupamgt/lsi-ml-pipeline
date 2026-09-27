"""Stage A VLM clients: Anthropic, Gemini, OpenAI, and local Ollama.

Temperature is fixed at 0. Every call is cached under ``outputs_vlm/cache/*.jsonl``
keyed by (tile hash, prompt hash, model). API keys come from environment
variables only — never written to disk.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from lsi_vlm.prompts import VLMPrediction, parse_vlm_response

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("outputs_vlm/cache")
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_S = 1.5

ENV_ANTHROPIC = "ANTHROPIC_API_KEY"
ENV_GEMINI = "GEMINI_API_KEY"
ENV_OPENAI = "OPENAI_API_KEY"

# Prefer GEMINI_API_KEY; accept GOOGLE_API_KEY as a common alias (read-only).
ENV_GOOGLE_ALIAS = "GOOGLE_API_KEY"


@dataclass(frozen=True)
class CacheKey:
    tile_hash: str
    prompt_hash: str
    model: str

    def as_dict(self) -> dict[str, str]:
        return {
            "tile_hash": self.tile_hash,
            "prompt_hash": self.prompt_hash,
            "model": self.model,
        }

    @property
    def key_id(self) -> str:
        raw = f"{self.tile_hash}|{self.prompt_hash}|{self.model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class VLMCallResult:
    """Outcome of a single VLM completion (cached or live)."""

    raw_text: str
    prediction: VLMPrediction | None
    backend: str
    model: str
    cached: bool
    cache_key: CacheKey
    latency_s: float | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_cache_record(self) -> dict[str, Any]:
        return {
            **self.cache_key.as_dict(),
            "key_id": self.cache_key.key_id,
            "backend": self.backend,
            "model": self.model,
            "raw_text": self.raw_text,
            "prediction": self.prediction.to_json_dict() if self.prediction else None,
            "error": self.error,
            "latency_s": self.latency_s,
            "meta": self.meta,
        }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_tile_payload(
    *parts: bytes | str | Path,
) -> str:
    """Stable tile hash from image bytes/paths and optional side strings."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, Path):
            h.update(part.read_bytes())
        elif isinstance(part, bytes):
            h.update(part)
        else:
            h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def prompt_hash(prompt: str) -> str:
    return sha256_text(prompt)


def _safe_model_filename(model: str) -> str:
    return re_sub_non_alnum(model)


def re_sub_non_alnum(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def cache_path_for(model: str, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> Path:
    """One JSONL file per model under the cache directory."""
    return Path(cache_dir) / f"{_safe_model_filename(model)}.jsonl"


# ---------------------------------------------------------------------------
# JSONL cache helpers
# ---------------------------------------------------------------------------


def ensure_cache_dir(cache_dir: Path | str = DEFAULT_CACHE_DIR) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_cache_index(cache_file: Path) -> dict[str, dict[str, Any]]:
    """Load key_id → record from a JSONL cache file."""
    index: dict[str, dict[str, Any]] = {}
    if not cache_file.is_file():
        return index
    with cache_file.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt cache line %s:%d", cache_file, line_no)
                continue
            kid = rec.get("key_id")
            if not kid:
                th, ph, model = rec.get("tile_hash"), rec.get("prompt_hash"), rec.get("model")
                if th and ph and model:
                    kid = CacheKey(str(th), str(ph), str(model)).key_id
                else:
                    continue
            index[str(kid)] = rec
    return index


def append_cache_record(cache_file: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object as a line. Never writes API keys."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    # Defence: strip anything that looks like a secret.
    payload = dict(record)
    for bad in (
        "api_key",
        "API_KEY",
        "authorization",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        payload.pop(bad, None)
    with cache_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def lookup_cache(
    cache_dir: Path | str,
    key: CacheKey,
) -> dict[str, Any] | None:
    path = cache_path_for(key.model, cache_dir)
    index = load_cache_index(path)
    return index.get(key.key_id)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _read_image_bytes(image: bytes | str | Path) -> bytes:
    if isinstance(image, bytes):
        return image
    return Path(image).read_bytes()


def _guess_media_type(image: bytes | str | Path) -> str:
    name = ""
    if isinstance(image, (str, Path)):
        name = str(image).lower()
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        return "image/jpeg"
    if name.endswith(".webp"):
        return "image/webp"
    return "image/png"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# Abstract client
# ---------------------------------------------------------------------------


class VLMClient(ABC):
    """Common interface for Stage A vision-language backends."""

    backend: str = "base"

    def __init__(
        self,
        *,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        use_cache: bool = True,
    ) -> None:
        self.model = model
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        if use_cache:
            ensure_cache_dir(self.cache_dir)

    @abstractmethod
    def _complete_once(
        self,
        images: Sequence[bytes | str | Path],
        prompt: str,
    ) -> str:
        """Provider-specific single attempt; raise on transport/API errors."""

    def complete(
        self,
        images: Sequence[bytes | str | Path],
        prompt: str,
        *,
        tile_hash: str | None = None,
        parse: bool = True,
    ) -> VLMCallResult:
        """Run (or load) a completion. Cache key = (tile_hash, prompt_hash, model)."""
        ph = prompt_hash(prompt)
        if tile_hash is None:
            tile_hash = hash_tile_payload(*[_read_image_bytes(im) for im in images])
        key = CacheKey(tile_hash=tile_hash, prompt_hash=ph, model=self.model)

        if self.use_cache:
            hit = lookup_cache(self.cache_dir, key)
            if hit is not None:
                raw = str(hit.get("raw_text") or "")
                pred = None
                if parse:
                    if hit.get("prediction"):
                        try:
                            from lsi_vlm.prompts import validate_prediction

                            pred = validate_prediction(hit["prediction"])
                        except (ValueError, TypeError, KeyError):
                            pred = parse_vlm_response(raw)
                    else:
                        pred = parse_vlm_response(raw)
                return VLMCallResult(
                    raw_text=raw,
                    prediction=pred,
                    backend=self.backend,
                    model=self.model,
                    cached=True,
                    cache_key=key,
                    latency_s=None,
                    error=hit.get("error"),
                    meta={"source": "cache"},
                )

        last_err: Exception | None = None
        raw_text = ""
        t0 = time.perf_counter()
        for attempt in range(1, self.max_retries + 1):
            try:
                raw_text = self._complete_once(images, prompt)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001 — log + retry all provider errors
                last_err = exc
                logger.warning(
                    "%s attempt %d/%d failed for model=%s: %s",
                    self.backend,
                    attempt,
                    self.max_retries,
                    self.model,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * attempt)
        latency = time.perf_counter() - t0

        error_msg = str(last_err) if last_err else None
        if last_err and not raw_text:
            logger.error(
                "%s exhausted retries for model=%s: %s",
                self.backend,
                self.model,
                last_err,
            )

        pred = parse_vlm_response(raw_text) if (parse and raw_text) else None
        if raw_text and pred is None and parse:
            logger.error(
                "%s parse failure for model=%s (logged; not silently dropped)",
                self.backend,
                self.model,
            )

        result = VLMCallResult(
            raw_text=raw_text,
            prediction=pred,
            backend=self.backend,
            model=self.model,
            cached=False,
            cache_key=key,
            latency_s=latency,
            error=error_msg,
            meta={"attempts": self.max_retries if last_err else 1},
        )
        if self.use_cache and (raw_text or error_msg):
            append_cache_record(cache_path_for(self.model, self.cache_dir), result.to_cache_record())
        return result


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _require_env(var: str, *aliases: str) -> str:
    for name in (var, *aliases):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    names = ", ".join((var, *aliases))
    raise RuntimeError(
        f"Missing API key env var(s): {names}. Set the key in the environment; "
        "do not write secrets to files."
    )


class AnthropicClient(VLMClient):
    backend = "anthropic"

    def __init__(self, *, model: str = "claude-sonnet-4-20250514", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._api_key = _require_env(ENV_ANTHROPIC)

    def _complete_once(self, images: Sequence[bytes | str | Path], prompt: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for the Anthropic backend. "
                "Install with: pip install -e '.[vlm]'"
            ) from exc

        client = anthropic.Anthropic(api_key=self._api_key)
        content: list[dict[str, Any]] = []
        for image in images:
            data = _read_image_bytes(image)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _guess_media_type(image),
                        "data": _b64(data),
                    },
                }
            )
        content.append({"type": "text", "text": prompt})
        msg = client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )
        parts = []
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()


class GeminiClient(VLMClient):
    backend = "gemini"

    def __init__(self, *, model: str = "gemini-2.0-flash", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._api_key = _require_env(ENV_GEMINI, ENV_GOOGLE_ALIAS)

    def _complete_once(self, images: Sequence[bytes | str | Path], prompt: str) -> str:
        # Prefer the newer google-genai package; fall back to google-generativeai.
        try:
            return self._complete_google_genai(images, prompt)
        except ImportError:
            return self._complete_generativeai(images, prompt)

    def _complete_google_genai(
        self, images: Sequence[bytes | str | Path], prompt: str
    ) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        parts: list[Any] = [prompt]
        for image in images:
            data = _read_image_bytes(image)
            parts.append(
                types.Part.from_bytes(data=data, mime_type=_guess_media_type(image))
            )
        response = client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(temperature=self.temperature),
        )
        text = getattr(response, "text", None) or ""
        return str(text).strip()

    def _complete_generativeai(
        self, images: Sequence[bytes | str | Path], prompt: str
    ) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-genai or google-generativeai is required for Gemini. "
                "Install with: pip install -e '.[vlm]'"
            ) from exc

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self.model)
        parts: list[Any] = [prompt]
        for image in images:
            data = _read_image_bytes(image)
            parts.append({"mime_type": _guess_media_type(image), "data": data})
        response = model.generate_content(
            parts,
            generation_config={"temperature": self.temperature},
        )
        text = getattr(response, "text", None) or ""
        return str(text).strip()


class OpenAIClient(VLMClient):
    backend = "openai"

    def __init__(self, *, model: str = "gpt-4o", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._api_key = _require_env(ENV_OPENAI)

    def _complete_once(self, images: Sequence[bytes | str | Path], prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for the OpenAI backend. "
                "Install with: pip install -e '.[vlm]'"
            ) from exc

        client = OpenAI(api_key=self._api_key)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            data = _read_image_bytes(image)
            media = _guess_media_type(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media};base64,{_b64(data)}"},
                }
            )
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
        )
        choice = resp.choices[0].message
        return (choice.content or "").strip()


class OllamaClient(VLMClient):
    """Local Ollama vision backend (e.g. ``qwen2.5vl``). No API key required."""

    backend = "ollama"

    def __init__(
        self,
        *,
        model: str = "qwen2.5vl",
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _complete_once(self, images: Sequence[bytes | str | Path], prompt: str) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for the Ollama backend. "
                "Install with: pip install -e '.[vlm]'"
            ) from exc

        images_b64 = [_b64(_read_image_bytes(im)) for im in images]
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        url = f"{self.base_url}/api/chat"
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        message = data.get("message") or {}
        content = message.get("content")
        if content is None:
            # Some models return top-level ``response``
            content = data.get("response", "")
        return str(content or "").strip()


BACKENDS = {
    "anthropic": AnthropicClient,
    "claude": AnthropicClient,
    "gemini": GeminiClient,
    "google": GeminiClient,
    "openai": OpenAIClient,
    "ollama": OllamaClient,
}


def make_client(
    backend: str,
    *,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    **kwargs: Any,
) -> VLMClient:
    """Factory for Stage A VLM backends.

    Parameters
    ----------
    backend:
        One of ``anthropic``, ``gemini``, ``openai``, ``ollama``.
    model:
        Provider model id (defaults differ per backend).
    """
    key = backend.strip().lower()
    if key not in BACKENDS:
        raise ValueError(
            f"Unknown VLM backend {backend!r}. Choose from: {sorted(set(BACKENDS))}"
        )
    cls = BACKENDS[key]
    init_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_retries": max_retries,
        "cache_dir": cache_dir,
        "use_cache": use_cache,
        **kwargs,
    }
    if model is not None:
        init_kwargs["model"] = model
    return cls(**init_kwargs)


def client_from_config(cfg: Mapping[str, Any]) -> VLMClient:
    """Build a client from the ``model`` section of ``configs/vlm_stage_a.yaml``."""
    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else cfg
    backend = str(model_cfg.get("backend", "ollama"))
    kwargs: dict[str, Any] = {
        "model": model_cfg.get("name") or model_cfg.get("model"),
        "temperature": float(model_cfg.get("temperature", DEFAULT_TEMPERATURE)),
        "max_retries": int(model_cfg.get("max_retries", DEFAULT_MAX_RETRIES)),
        "cache_dir": model_cfg.get("cache_dir", DEFAULT_CACHE_DIR),
        "use_cache": bool(model_cfg.get("use_cache", True)),
    }
    # Drop None model so backend defaults apply
    if kwargs["model"] is None:
        kwargs.pop("model")
    if backend.lower() in ("ollama",) and model_cfg.get("ollama_base_url"):
        kwargs["base_url"] = model_cfg["ollama_base_url"]
    if model_cfg.get("timeout_s") is not None:
        kwargs["timeout_s"] = float(model_cfg["timeout_s"])
    return make_client(backend, **kwargs)
