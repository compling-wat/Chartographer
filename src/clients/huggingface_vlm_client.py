# huggingface_vlm_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                          AutoProcessor, AutoTokenizer,
                          LlavaOnevisionForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration,
                          Qwen3VLForConditionalGeneration)

from config.model_alias_resolver import ModelResolver

Message = Dict[str, Any]

VLM_HINT_TAGS = ("llava-ov", "internvl", "pixtral", "qwen2.5-vl", "qwen3-vl", "gemma4")
EAGER_ATTN_TAGS = ("pixtral", "gemma4")


@dataclass
class ClientConfig:
    model: str
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps" | "cuda:0"
    dtype: str = "auto"   # "auto" | "float16" | "bfloat16" | "float32"
    max_new_tokens: int = 1024
    trust_remote_code: bool = True


def _to_torch_dtype(name: str):
    name = (name or "auto").lower()
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": None,
    }.get(name, None)


def _pick_device(pref: str):
    pref = (pref or "auto").lower()

    # AUTO -> let HF shard across GPUs using device_map="auto"
    if pref == "auto":
        return None

    if pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(pref)

    if pref.startswith("mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _ensure_padding_eos(tok_like) -> None:
    if tok_like is None:
        return
    try:
        tok_like.padding_side = "left"  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        if getattr(tok_like, "pad_token", None) is None and getattr(tok_like, "eos_token", None) is not None:
            tok_like.pad_token = tok_like.eos_token  # type: ignore[attr-defined]
    except Exception:
        pass


def _ensure_model_padid(model, tok_like) -> None:
    try:
        gc = getattr(model, "generation_config", None)
        if gc is not None and getattr(gc, "pad_token_id", None) is None:
            pad_id = getattr(tok_like, "pad_token_id", None) or getattr(tok_like, "eos_token_id", None)
            if pad_id is not None:
                gc.pad_token_id = pad_id
    except Exception:
        pass


def _move_to_device(
    obj: Dict[str, Any],
    device: Optional[torch.device],
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    if device is None:
        return obj
    moved: Dict[str, Any] = {}
    for k, v in obj.items():
        if not hasattr(v, "to"):
            moved[k] = v
            continue

        # Keep integer/bool tensors (e.g., input_ids/attention_mask) unchanged;
        # cast floating tensors (e.g., pixel_values) to match model weights.
        if torch.is_tensor(v) and v.is_floating_point() and dtype is not None:
            moved[k] = v.to(device=device, dtype=dtype)
        else:
            moved[k] = v.to(device)
    return moved


def _run_generate(
    model,
    inputs: Dict[str, Any],
    pad_id: Optional[int],
    tok_like,
    max_new_tokens: Optional[int] = None,
) -> List[str]:
    generate_kwargs: Dict[str, Any] = {"pad_token_id": pad_id}
    if max_new_tokens is not None:
        generate_kwargs["max_new_tokens"] = max_new_tokens
    gen = model.generate(**inputs, **generate_kwargs)
    gen_cpu = gen.detach().cpu()
    gen_to_decode = gen_cpu

    input_ids = inputs.get("input_ids")
    if torch.is_tensor(input_ids):
        in_cpu = input_ids.detach().cpu()
        if (
            in_cpu.ndim == 2
            and gen_cpu.ndim == 2
            and gen_cpu.shape[0] == in_cpu.shape[0]
            and gen_cpu.shape[1] >= in_cpu.shape[1]
        ):
            prompt_len = int(in_cpu.shape[1])
            gen_to_decode = gen_cpu[:, prompt_len:]

    if hasattr(tok_like, "batch_decode"):
        return [s.strip() for s in tok_like.batch_decode(gen_to_decode, skip_special_tokens=True)]
    if hasattr(tok_like, "decode"):
        return [tok_like.decode(t, skip_special_tokens=True).strip() for t in gen_to_decode]
    return [str(t) for t in gen_to_decode]


def _normalize_vlm_messages(messages: List[Message], images: Optional[Sequence[Any]]) -> List[Message]:
    """Convert to: content = [{type: text|image, ...}, ...], and append images to last user msg."""
    out: List[Message] = []
    for m in messages or []:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            segs = list(content)
        else:
            segs = [{"type": "text", "text": str(content)}]
        out.append({"role": role, "content": segs})

    imgs = list(images or [])
    if imgs:
        target: Optional[Message] = None
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                target = out[i]
                break
        if target is None:
            target = {"role": "user", "content": []}
            out.append(target)
        for img in imgs:
            target["content"].append({"type": "image", "image": img})
    return out


def _render_chat_fallback(messages: List[Message], *, add_generation_prompt: bool = True) -> str:
    def _get_text_content(m: Message) -> str:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(seg.get("text", "") for seg in content if seg.get("type") == "text")
        return str(content)

    buf: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = _get_text_content(m)
        prefix = "System" if role == "system" else ("Assistant" if role == "assistant" else "User")
        buf.append(f"{prefix}: {content}")
    if add_generation_prompt:
        buf.append("Assistant:")
    return "\n".join(buf)


def _merge_system_into_first_user(messages: List[Message]) -> List[Message]:
    """Merge all system messages into the first user message as plain text."""
    if not messages:
        return messages

    system_texts: List[str] = []
    new_msgs: List[Message] = []

    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            content = m.get("content", "")
            if isinstance(content, list):
                text = "".join(
                    seg.get("text", "")
                    for seg in content
                    if isinstance(seg, dict) and seg.get("type") == "text"
                )
            else:
                text = str(content)
            text = text.strip()
            if text:
                system_texts.append(text)
        else:
            new_msgs.append(m)

    if not system_texts:
        return messages

    prefix = "\n".join(system_texts)
    if new_msgs and new_msgs[0].get("role") == "user":
        content = new_msgs[0].get("content", "")
        if isinstance(content, list):
            new_msgs[0]["content"] = [{"type": "text", "text": prefix + "\n"}] + content
        else:
            new_msgs[0]["content"] = prefix + "\n" + str(content)
    else:
        new_msgs.insert(0, {"role": "user", "content": prefix})

    return new_msgs


class ModelClient:
    """Lightweight generation wrapper for text-only and VLM Hugging Face models."""

    debug_print_budget: int = 0

    @classmethod
    def reset_debug(cls, budget: int = 1) -> None:
        cls.debug_print_budget = int(budget)

    def __init__(self, cfg: ClientConfig) -> None:
        if AutoModelForCausalLM is None and AutoModelForImageTextToText is None:
            raise RuntimeError("transformers is not available")

        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self.dtype = _to_torch_dtype(cfg.dtype)
        self.max_new_tokens = int(cfg.max_new_tokens) if cfg.max_new_tokens is not None else None

        # --- resolve alias -> local weights or HF repo id ---
        resolver = ModelResolver()
        repo = resolver.resolve(cfg.model)

        # Keep model_key as the *original* identifier for cleanup heuristics
        self.model_key = (cfg.model or "").lower()
        self.resolved_repo = repo
        repo_lc = str(repo).lower()

        load_kwargs: Dict[str, Any] = {"trust_remote_code": cfg.trust_remote_code}

        if self.dtype is not None:
            load_kwargs["torch_dtype"] = self.dtype
            print(f"[Loading model with dtype: {self.dtype}]")

        if self.device is None:
            load_kwargs["device_map"] = "auto"
            print("[Device mode: auto: using all available GPUs with device_map='auto']")
        elif self.device.type == "cuda":
            load_kwargs["device_map"] = None
            print(f"[Device mode: CUDA: using device {self.device}]")
        elif self.device.type == "mps":
            load_kwargs["device_map"] = None
            print("[Device mode: MPS]")
        else:
            load_kwargs["device_map"] = None
            print("[Device mode: CPU]")

        # Some model families are brittle with FlashAttention depending on
        # transformers/flash-attn versions; force eager attention for stability.
        if any(tag in repo_lc or tag in self.model_key for tag in EAGER_ATTN_TAGS):
            load_kwargs["attn_implementation"] = "eager"
            print("[Attention backend: eager (flash-attn disabled for this model)]")
        else:
            load_kwargs["attn_implementation"] = "flash_attention_2"  # for better speed where supported

        print(f"[Loading model repo: {repo}]")

        self.model = None
        self.tokenizer = None
        self.processor = None

        # Treat as VLM if the string suggests VLM (alias, HF id, or local dir name)
        looks_vlm = any(
            tag in repo_lc or tag in self.model_key
            for tag in VLM_HINT_TAGS
        )

        # Text-only first (unless explicitly/implicitly VLM).
        if not looks_vlm:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    repo, trust_remote_code=cfg.trust_remote_code, use_fast=True
                )
                _ensure_padding_eos(self.tokenizer)
                self.model = AutoModelForCausalLM.from_pretrained(repo, **load_kwargs)
                if load_kwargs["device_map"] is None:
                    self.model.to(self.device)
                _ensure_model_padid(self.model, self.tokenizer)
            except Exception:
                self.tokenizer = None
                self.model = None

        # VLM path (or fallback if text load failed).
        if self.model is None and AutoModelForImageTextToText is not None and AutoProcessor is not None:
            self.processor = AutoProcessor.from_pretrained(
                repo, trust_remote_code=cfg.trust_remote_code, use_fast=True
            )
            tok = getattr(self.processor, "tokenizer", None)
            _ensure_padding_eos(tok)

            # Use repo/model_key hints to pick specialized classes where needed
            if "llava" in repo_lc or "llava" in self.model_key:
                self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(repo, **load_kwargs)
            elif ("qwen2.5-vl" in repo_lc) or ("qwen2.5-vl" in self.model_key):
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo, **load_kwargs)
            elif ("qwen3-vl" in repo_lc) or ("qwen3-vl" in self.model_key):
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(repo, **load_kwargs)
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(repo, **load_kwargs)

            if load_kwargs["device_map"] is None:
                self.model.to(self.device)

        if self.model is None:
            raise RuntimeError(f"Could not load model for: {cfg.model} (resolved: {repo})")

        # Infer concrete device when device_map='auto'
        if self.device is None:
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.model.eval()

    # ---------------- debug + extraction ----------------

    def _maybe_debug_print(self, *, prompt: str, output: str, kind: str) -> None:
        if self.__class__.debug_print_budget <= 0:
            return
        print(f"\n================ DEBUG: MODEL INPUT ({kind}) ================\n")
        print(prompt)
        print(f"\n================ DEBUG: MODEL OUTPUT ({kind}) ===============\n")
        print(output)
        print("\n=============================================================\n")
        self.__class__.debug_print_budget -= 1

    # ---------------- unified generation core ----------------

    def _generate_items(self, items: Sequence[Dict[str, Any]]) -> List[str]:
        if not items:
            return []

        debug_prompts: List[str] = []
        key = (self.model_key or "").lower()
        is_mistral_family = "mistral" in key and "pixtral" not in key
        is_pixtral_family = "pixtral" in key

        if self.processor is not None:  # VLM
            tok = getattr(self.processor, "tokenizer", None)

            if not hasattr(self.processor, "apply_chat_template"):
                raise RuntimeError("VLM processor missing apply_chat_template; no manual fallback implemented.")

            conversations: List[List[Message]] = []
            for it in items:
                conv = _normalize_vlm_messages(it.get("messages", []), it.get("images"))
                if is_pixtral_family:
                    conv = _merge_system_into_first_user(conv)
                conversations.append(conv)

            for conv in conversations:
                dp = self.processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)
                debug_prompts.append(dp)

            # Newer transformers expects processor call kwargs to be nested under
            # processor_kwargs; older builds may still accept direct kwargs.
            try:
                inputs = self.processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    processor_kwargs={"return_tensors": "pt", "padding": True},
                )
            except TypeError:
                inputs = self.processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    return_dict=True,
                )
            inputs = _move_to_device(inputs, self.device, self.dtype)
            pad_id = getattr(tok, "eos_token_id", None) if tok else None
            decoded = _run_generate(
                self.model,
                inputs,
                pad_id,
                tok or self.processor,
                max_new_tokens=self.max_new_tokens,
            )

        else:  # text-only
            assert self.tokenizer is not None

            if hasattr(self.tokenizer, "apply_chat_template"):
                conversations = [it.get("messages", []) for it in items]
                if is_mistral_family:
                    conversations = [_merge_system_into_first_user(conv) for conv in conversations]

                for conv in conversations:
                    dp = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                    debug_prompts.append(dp)

                inputs = self.tokenizer.apply_chat_template(
                    conversations,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    return_dict=True,
                )
            else:
                prompts: List[str] = [
                    _render_chat_fallback(it.get("messages", []), add_generation_prompt=True)
                    for it in items
                ]
                debug_prompts = prompts
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)

            inputs = _move_to_device(inputs, self.device, self.dtype)
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
            decoded = _run_generate(
                self.model,
                inputs,
                pad_id,
                self.tokenizer,
                max_new_tokens=self.max_new_tokens,
            )

        cleaned = [d.strip() for d in decoded]

        if self.__class__.debug_print_budget > 0 and debug_prompts and cleaned:
            kind = "vlm-batch" if self.processor is not None else "text-batch"
            self._maybe_debug_print(prompt=debug_prompts[0], output=cleaned[0], kind=kind)

        return cleaned

    # ---------------- public API ----------------

    @torch.inference_mode()
    def generate(self, messages: List[Message], images: Optional[Sequence[Any]] = None) -> str:
        items = [{"messages": messages, "images": images}]
        results = self._generate_items(items)
        return results[0] if results else ""

    @torch.inference_mode()
    def generate_batch(self, items: Sequence[Dict[str, Any]]) -> List[str]:
        return self._generate_items(items)
