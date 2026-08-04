"""Локальный held-out eval HRM-Text-1B на BIM-задачах (protocol / success)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

from bim_gym.eval import HeldoutEvalConfig, metrics_to_dict, run_heldout
from bim_gym.hrm import DEFAULT_MODEL_ID, generate_from_messages

GATE_PROTOCOL = 0.90
GATE_SUCCESS = 0.55
GATE_UNKNOWN = 0.02


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIM held-out eval for HRM-Text-1B")
    p.add_argument("--model", default=os.environ.get("BIM_GYM_MODEL", DEFAULT_MODEL_ID))
    p.add_argument("--lora", default=os.environ.get("BIM_GYM_LORA"), help="optional PEFT adapter dir")
    p.add_argument("--eval-seed", type=int, default=int(os.environ.get("BIM_GYM_EVAL_SEED", "10000")))
    p.add_argument("--train-seed", type=int, default=int(os.environ.get("BIM_GYM_TRAIN_SEED", "0")))
    p.add_argument("--n-tasks", type=int, default=int(os.environ.get("BIM_GYM_EVAL_N", "16")))
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("BIM_GYM_MAX_NEW", "1024")))
    p.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("BIM_GYM_HELDOUT_OUT", r"c:\dev\BIM-Gym\out\local_hrm_heldout.json")),
    )
    p.add_argument("--show", type=int, default=1, help="print N sample generations")
    return p.parse_args()


def load_model(model_id: str, lora: str | None):
    tok_src = lora if lora and (Path(lora) / "tokenizer_config.json").exists() else model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    if lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))

    print("loading", args.model, "lora", args.lora or "(none)")
    model, tokenizer = load_model(args.model, args.lora)
    if torch.cuda.is_available():
        print("alloc_gb", round(torch.cuda.memory_allocated() / 1024**3, 2))

    samples: list[dict] = []

    @torch.inference_mode()
    def generate_fn(messages: list[dict[str, str]]) -> str:
        text = generate_from_messages(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=0.2,
            do_sample=True,
        )
        if len(samples) < args.show:
            samples.append({"user": messages[-1]["content"][:400], "assistant": text[:800]})
        return text

    cfg = HeldoutEvalConfig(
        seed=args.eval_seed,
        train_seed=args.train_seed,
        n_tasks=args.n_tasks,
    )
    print("heldout", {"seed": cfg.seed, "train_seed": cfg.train_seed, "n_tasks": cfg.n_tasks})
    report = run_heldout(cfg, generate_fn=generate_fn)
    metrics = metrics_to_dict(report.metrics)
    gates = {
        "protocol": metrics["protocol_valid_rate"] >= GATE_PROTOCOL,
        "success": metrics["success_rate"] >= GATE_SUCCESS,
        "unknown": metrics["unknown_primitive_rate"] <= GATE_UNKNOWN,
    }
    payload = {
        "model": args.model,
        "lora": args.lora,
        "config": {"eval_seed": cfg.seed, "train_seed": cfg.train_seed, "n_tasks": cfg.n_tasks},
        "thresholds": {
            "protocol_valid": GATE_PROTOCOL,
            "success": GATE_SUCCESS,
            "unknown_primitive": GATE_UNKNOWN,
        },
        "metrics": metrics,
        "gates": gates,
        "samples": samples,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "gates": gates}, ensure_ascii=False, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
