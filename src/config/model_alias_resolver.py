# model_resolver.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def _non_empty_path(path: str) -> bool:
    """Return True if `path` exists and is a non-empty file or directory."""
    try:
        p = Path(path)
        if p.is_file():
            return True
        if p.is_dir():
            try:
                return any(p.iterdir())
            except Exception:
                return False
        return False
    except Exception:
        return False


class ModelResolver:
    """Resolve aliases to local or Hugging Face repositories.

    Resolution rules:
    1) If given a real filesystem path, return as-is.
    2) If given a known alias, optionally map to a local path under
       `CHARTOGRAPHER_MODEL_WEIGHTS_DIR`.
    3) If the local path does not exist, fall back to the HF repo ID.
    4) Otherwise, treat the string as an HF repo ID or arbitrary path.
    """

    def __init__(self, model_weights_dir: Optional[str] = None) -> None:
        self.model_weights_dir = model_weights_dir or os.getenv("CHARTOGRAPHER_MODEL_WEIGHTS_DIR", "")

        self.alias_map: Dict[str, str] = {
            "llava-ov": "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
            "pixtral": "mistral-community/pixtral-12b",
            "qwen2.5-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
            "internvl3": "OpenGVLab/InternVL3-8B-hf",
            "gemma4": "google/gemma-4-E4B-it",
            "qwen3-vl": "Qwen/Qwen3-VL-8B-Instruct",
        }

    def _repo_tail(self, hf_id: str) -> str:
        return hf_id.rsplit("/", 1)[-1]

    def resolve(self, model_or_alias: str) -> str:
        """Resolve `model_or_alias` to a concrete local path or HF repo ID."""
        if not model_or_alias:
            raise ValueError("model_or_alias is empty")

        # 1) Existing path?
        if _non_empty_path(model_or_alias):
            return model_or_alias

        key = model_or_alias.lower()

        # 2) Known alias -> prefer local weights -> else HF id
        if key in self.alias_map:
            hf_id = self.alias_map[key]
            tail = self._repo_tail(hf_id)

            if self.model_weights_dir:
                local_path = Path(self.model_weights_dir) / tail
                if _non_empty_path(str(local_path)):
                    print(f"[Resolved {model_or_alias} -> local: {local_path}]")
                    return str(local_path)

            print(f"[Local weights not found, falling back to HF: {hf_id}]")
            return hf_id

        # 3) Otherwise assume it is an HF ID or non-existing path-like string
        return model_or_alias
