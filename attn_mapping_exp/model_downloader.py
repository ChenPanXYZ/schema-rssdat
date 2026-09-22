import os
from typing import Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_CACHE_DIR = ".cache"

def _get_hf_token() -> Optional[str]:
    # Support common env var names
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )


def _from_pretrained_with_token(cls, model_name: str, **kwargs):
    """
    Transformers has used both `token=` and `use_auth_token=` across versions.
    Use `token=` first; fallback to `use_auth_token=` if needed.
    """
    token = _get_hf_token()
    if token:
        try:
            return cls.from_pretrained(model_name, token=token, cache_dir=HF_CACHE_DIR, **kwargs)
        except TypeError:
            return cls.from_pretrained(model_name, use_auth_token=token, cache_dir=HF_CACHE_DIR, **kwargs)
    return cls.from_pretrained(model_name, cache_dir=HF_CACHE_DIR, **kwargs)


def download_model(
    model_name: str,
    *,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Download/load a HF causal LM + tokenizer for attention visualization.

    Returns (model, tokenizer) and places the model on a single device
    ("cuda" if available, else "cpu") so downstream code can simply do
    `inputs.to(model.device)` / `inputs.to("cuda")`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch_dtype is None:
        if device == "cuda":
            # Prefer bf16 if available; fallback to fp16.
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32

    tokenizer = _from_pretrained_with_token(AutoTokenizer, model_name, use_fast=True)

    # Some LLaMA-like models may not have pad_token set.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _from_pretrained_with_token(
        AutoModelForCausalLM,
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    model.eval()
    model.to(device)
    return model, tokenizer

