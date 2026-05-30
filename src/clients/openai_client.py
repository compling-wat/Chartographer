from __future__ import annotations

import base64
import io
import os
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import google.auth.transport.requests
from google.auth import default
from openai import OpenAI
from PIL import Image

Message = Dict[str, Any]


@dataclass
class OpenAIClientConfig:
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_sec: float = 120.0
    max_retries: int = 5
    retry_base_delay_sec: float = 1.0
    retry_max_delay_sec: float = 30.0


def _to_data_url(img: Any) -> Optional[str]:
    if img is None:
        return None

    if isinstance(img, str):
        if img.startswith("http://") or img.startswith("https://"):
            return img
        if os.path.exists(img):
            with open(img, "rb") as f:
                data = f.read()
        else:
            return None
    elif isinstance(img, (bytes, bytearray)):
        data = bytes(img)
    elif isinstance(img, dict):
        if img.get("bytes") is not None:
            data = bytes(img["bytes"])
        elif img.get("path") is not None:
            p = img["path"]
            if isinstance(p, str) and os.path.exists(p):
                with open(p, "rb") as f:
                    data = f.read()
            else:
                return None
        else:
            return None
    elif Image is not None and isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
    else:
        return None

    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _append_images_to_last_user(messages: List[Message], images: Optional[Sequence[Any]]) -> List[Message]:
    if not images:
        return messages

    out = [dict(m) for m in messages]
    target = None
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            target = out[i]
            break

    if target is None:
        target = {"role": "user", "content": []}
        out.append(target)

    content = target.get("content", "")
    if isinstance(content, list):
        segments = list(content)
    else:
        segments = [{"type": "text", "text": str(content)}]

    for img in images:
        url = _to_data_url(img)
        if url:
            segments.append({"type": "image_url", "image_url": {"url": url}})

    target["content"] = segments
    return out


def _normalize_messages(messages: List[Message]) -> List[Message]:
    out: List[Message] = []
    for m in messages or []:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            segments = []
            for seg in content:
                if seg.get("type") == "image":
                    url = _to_data_url(seg.get("image"))
                    if url:
                        segments.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    segments.append({"type": "text", "text": str(seg.get("text", ""))})
            out.append({"role": role, "content": segments})
        else:
            out.append({"role": role, "content": str(content)})
    return out


class OpenAIClient:
    def __init__(self, cfg: OpenAIClientConfig) -> None:
        model_lc = (cfg.model or "").lower()
        self._is_gemini = "gemini" in model_lc
        self._lock = threading.Lock()
        self._creds = None
        self._request = None

        api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
        base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL")

        if self._is_gemini and not base_url:
            project_id = (os.getenv("GOOGLE_CLOUD_PROJECT") or "chartographer").strip()
            location = (os.getenv("GOOGLE_CLOUD_LOCATION") or "northamerica-northeast1").strip()
            base_url = (
                f"https://{location}-aiplatform.googleapis.com/v1/projects/"
                f"{project_id}/locations/{location}/endpoints/openapi"
            )

        # Vertex OpenAPI expects publisher-prefixed model IDs: <publisher>/<model>.
        # Accept plain gemini names from CLI and normalize them automatically.
        if self._is_gemini and "/" not in (cfg.model or ""):
            cfg.model = f"google/{cfg.model}"

        # For Gemini on Vertex, OPENAI_API_KEY is often set for OpenAI models (sk-...).
        # That token type is invalid for Vertex and causes 401 ACCESS_TOKEN_TYPE_UNSUPPORTED.
        # In that case, fall back to ADC OAuth credentials instead.
        if self._is_gemini and isinstance(api_key, str) and api_key.startswith("sk-"):
            api_key = None

        if self._is_gemini and not api_key:
            creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            request = google.auth.transport.requests.Request()
            creds.refresh(request)
            api_key = creds.token
            self._creds = creds
            self._request = request

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.cfg = cfg
        self._base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._max_retries = max(0, int(cfg.max_retries))
        self._retry_base_delay = max(0.0, float(cfg.retry_base_delay_sec))
        self._retry_max_delay = max(self._retry_base_delay, float(cfg.retry_max_delay_sec))
        self._timeout_sec = max(1.0, float(cfg.timeout_sec))

    @staticmethod
    def _is_retriable_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and (status_code == 429 or 500 <= status_code < 600):
            return True

        msg = str(exc).lower()
        transient_markers = (
            "rate limit",
            "resource_exhausted",
            "too many requests",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "service unavailable",
            "internal error",
            "gateway timeout",
            "bad gateway",
        )
        if any(m in msg for m in transient_markers):
            return True

        name = exc.__class__.__name__.lower()
        if "timeout" in name or "connection" in name:
            return True
        return False

    def generate(
        self,
        messages: List[Message],
        images: Optional[Sequence[Any]] = None,
    ) -> str:
        if self._is_gemini and self._creds is not None and self._request is not None:
            with self._lock:
                expiry = getattr(self._creds, "expiry", None)
                token = getattr(self._creds, "token", None)
                needs_refresh = token is None
                if not needs_refresh and expiry is not None:
                    now = datetime.now(timezone.utc)
                    if expiry.tzinfo is None:
                        expiry = expiry.replace(tzinfo=timezone.utc)
                    needs_refresh = (expiry - now) <= timedelta(seconds=60)
                if needs_refresh:
                    self._creds.refresh(self._request)
                    self.client = OpenAI(api_key=self._creds.token, base_url=self._base_url)

        payload_messages = _append_images_to_last_user(_normalize_messages(messages), images)

        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=payload_messages,
                    timeout=self._timeout_sec,
                )
                choice = response.choices[0]
                return (choice.message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= self._max_retries or not self._is_retriable_error(e):
                    raise
                backoff = min(self._retry_max_delay, self._retry_base_delay * (2 ** attempt))
                jitter = random.uniform(0.0, max(0.001, backoff * 0.25))
                time.sleep(backoff + jitter)

        if last_err is not None:
            raise last_err
        raise RuntimeError("OpenAIClient.generate failed without a captured exception")

    def generate_batch(self, items: Sequence[Dict[str, Any]]) -> List[str]:
        outputs: List[str] = []
        for it in items:
            outputs.append(
                self.generate(
                    messages=it.get("messages", []),
                    images=it.get("images"),
                )
            )
        return outputs
