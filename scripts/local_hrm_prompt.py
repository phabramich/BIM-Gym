"""Локальный прогон одного промпта через HRM-Text-1B (PrefixLM)."""
from __future__ import annotations

import argparse
import os
import sys

os.environ.pop("BNB_ROCM_VERSION", None)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from bim_gym.hrm.rocm_compat import (  # noqa: E402
    install_rocm_distributed_stubs,
    patch_transformers_continuous_batching,
)

install_rocm_distributed_stubs()
patch_transformers_continuous_batching()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bim_gym.hrm import (
    DEFAULT_MODEL_ID,
    HrmConditionKind,
    generate_from_messages,
    messages_to_instruction,
)

DEFAULT_PROMPT = "Напиши куплет и припев песни в стиле OQJAV"
DEFAULT_SYSTEM = "You are a helpful assistant."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate from HRM-Text-1B with one prompt")
    p.add_argument("prompt", nargs="?", default=os.environ.get("BIM_GYM_PROMPT", DEFAULT_PROMPT))
    p.add_argument("--system", default=os.environ.get("BIM_GYM_SYSTEM", DEFAULT_SYSTEM))
    p.add_argument("--model", default=os.environ.get("BIM_GYM_MODEL", DEFAULT_MODEL_ID))
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("BIM_GYM_MAX_NEW", "512")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("BIM_GYM_TEMP", "0.8")))
    p.add_argument(
        "--condition",
        choices=[k.value for k in HrmConditionKind],
        default=os.environ.get("BIM_GYM_CONDITION", HrmConditionKind.Reasoning.value),
        help="reasoning = synth,cot (instruct-like); plain = empty condition",
    )
    p.add_argument("--greedy", action="store_true", help="do_sample=False")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    condition = HrmConditionKind(args.condition)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))

    print("loading", args.model, "...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map='auto' на ROCm Windows часто падает (0xC0000005)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    if torch.cuda.is_available():
        print("alloc_gb", round(torch.cuda.memory_allocated() / 1024**3, 2))

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    print("--- instruction ---")
    print(messages_to_instruction(messages, condition=condition))
    print("--- generate ---")
    text = generate_from_messages(
        model,
        tokenizer,
        messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=not args.greedy,
        condition=condition,
    )
    print(text)
    print("--- end ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
