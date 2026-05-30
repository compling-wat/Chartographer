from __future__ import annotations

import base64
import io
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from anthropic import Anthropic

from PIL import Image

Message = Dict[str, Any]


@dataclass
class ClaudeClientConfig:
    model: str
    api_key: Optional[str] = None
    max_tokens: int = 1024
    enable_prompt_caching: bool = True
    cache_ttl: Optional[str] = None


def _image_to_bytes_and_mime(img: Any) -> Optional[Tuple[bytes, str]]:
    if img is None:
        return None

    if isinstance(img, str):
        p = Path(img)
        if p.exists():
            data = p.read_bytes()
            mime = mimetypes.guess_type(str(p))[0] or "image/png"
            return data, mime
        return None

    if isinstance(img, (bytes, bytearray)):
        return bytes(img), "image/png"

    if isinstance(img, dict):
        if img.get("bytes") is not None:
            return bytes(img["bytes"]), "image/png"
        p = img.get("path")
        if isinstance(p, str) and Path(p).exists():
            data = Path(p).read_bytes()
            mime = mimetypes.guess_type(p)[0] or "image/png"
            return data, mime
        return None

    if isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"

    return None


def _normalize_messages(messages: List[Message]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    system_blocks: List[Dict[str, Any]] = []
    out: List[Dict[str, Any]] = []

    for m in messages or []:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            if isinstance(content, list):
                txt = "".join(str(seg.get("text", "")) for seg in content if seg.get("type") == "text")
            else:
                txt = str(content)
            txt = txt.strip()
            if txt:
                system_blocks.append({"type": "text", "text": txt})
            continue

        msg_role = "assistant" if role == "assistant" else "user"
        if isinstance(content, list):
            blocks: List[Dict[str, Any]] = []
            for seg in content:
                if seg.get("type") == "image":
                    b = _image_to_bytes_and_mime(seg.get("image"))
                    if b is None:
                        continue
                    data, mime = b
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": base64.b64encode(data).decode("utf-8"),
                            },
                        }
                    )
                else:
                    blocks.append({"type": "text", "text": str(seg.get("text", ""))})
            out.append({"role": msg_role, "content": blocks or [{"type": "text", "text": ""}]})
        else:
            out.append({"role": msg_role, "content": [{"type": "text", "text": str(content)}]})

    return system_blocks, out


def _append_images_to_last_user(
    messages: List[Dict[str, Any]],
    images: Optional[Sequence[Any]],
    *,
    cache_images: bool = False,
    cache_ttl: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not images:
        return messages

    out = [dict(m) for m in messages]
    target_idx = None
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            target_idx = i
            break

    if target_idx is None:
        out.append({"role": "user", "content": [{"type": "text", "text": ""}]})
        target_idx = len(out) - 1

    target_content = out[target_idx].get("content", [])
    if not isinstance(target_content, list):
        target_content = [{"type": "text", "text": str(target_content)}]

    image_blocks: List[Dict[str, Any]] = []
    for img in images:
        b = _image_to_bytes_and_mime(img)
        if b is None:
            continue
        data, mime = b
        image_block: Dict[str, Any] = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": base64.b64encode(data).decode("utf-8"),
            },
        }
        if cache_images:
            cc: Dict[str, Any] = {"type": "ephemeral"}
            if cache_ttl:
                cc["ttl"] = cache_ttl
            image_block["cache_control"] = cc
        image_blocks.append(image_block)

    out[target_idx]["content"] = target_content + image_blocks
    return out


class ClaudeClient:
    def __init__(self, cfg: ClaudeClientConfig) -> None:
        api_key = cfg.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        self.cfg = cfg
        self.client = Anthropic(api_key=api_key)

    def _build_request_params(
        self,
        messages: List[Message],
        images: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        system_blocks, norm = _normalize_messages(messages or [])
        norm = _append_images_to_last_user(
            norm,
            images,
            cache_images=bool(self.cfg.enable_prompt_caching),
            cache_ttl=self.cfg.cache_ttl,
        )

        kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": norm,
            "max_tokens": int(self.cfg.max_tokens),
        }

        if system_blocks:
            kwargs["system"] = system_blocks

        if self.cfg.enable_prompt_caching:
            cc: Dict[str, Any] = {"type": "ephemeral"}
            if self.cfg.cache_ttl:
                cc["ttl"] = self.cfg.cache_ttl
            if system_blocks:
                system_blocks[-1]["cache_control"] = cc

        return kwargs

    def generate(
        self,
        messages: List[Message],
        images: Optional[Sequence[Any]] = None,
    ) -> str:
        kwargs = self._build_request_params(messages=messages, images=images)
        resp = self.client.messages.create(**kwargs)
        texts = [block.text for block in (resp.content or []) if getattr(block, "type", None) == "text"]
        return "\n".join(texts).strip()

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
