"""Локальный SFT bootstrap HRM-Text-1B: baseline heldout → LoRA train → final heldout."""
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
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from bim_gym.eval import HeldoutEvalConfig, metrics_to_dict, run_heldout
from bim_gym.hrm import (
    DEFAULT_MODEL_ID,
    LORA_TARGET_MODULES,
    PrefixLmCollator,
    generate_from_messages,
    tokenize_sft_example,
)
from bim_gym.sft import SftDatasetConfig, build_dataset

GATE_PROTOCOL = 0.90
GATE_SUCCESS = 0.55
GATE_UNKNOWN = 0.02


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local HRM SFT bootstrap with held-out gates")
    p.add_argument("--model", default=os.environ.get("BIM_GYM_MODEL", DEFAULT_MODEL_ID))
    p.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("BIM_GYM_SFT_OUT", r"c:\dev\BIM-Gym\out\local_sft_full")),
    )
    # Defaults ≈ Colab bootstrap; 8GB usually fits seq 2048 with grad checkpointing.
    p.add_argument("--train-n", type=int, default=int(os.environ.get("BIM_GYM_TRAIN_N", "2000")))
    p.add_argument("--train-seed", type=int, default=int(os.environ.get("BIM_GYM_TRAIN_SEED", "0")))
    p.add_argument("--eval-n", type=int, default=int(os.environ.get("BIM_GYM_EVAL_N", "32")))
    p.add_argument("--baseline-n", type=int, default=int(os.environ.get("BIM_GYM_BASELINE_N", "8")))
    p.add_argument("--eval-seed", type=int, default=int(os.environ.get("BIM_GYM_EVAL_SEED", "10000")))
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("BIM_GYM_MAX_STEPS", "100")))
    p.add_argument("--seq", type=int, default=int(os.environ.get("BIM_GYM_SEQ", "2048")))
    p.add_argument("--lora-r", type=int, default=int(os.environ.get("BIM_GYM_LORA_R", "16")))
    p.add_argument("--grad-accum", type=int, default=int(os.environ.get("BIM_GYM_GRAD_ACCUM", "8")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("BIM_GYM_LR", "2e-4")))
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("BIM_GYM_MAX_NEW", "512")))
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-final", action="store_true")
    return p.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", path)


def evaluate(
    *,
    tag: str,
    model,
    tokenizer,
    eval_seed: int,
    train_seed: int,
    n_tasks: int,
    max_new_tokens: int = 512,
) -> dict:
    print(f"{tag}: running heldout n={n_tasks} ...", flush=True)
    cfg = HeldoutEvalConfig(seed=eval_seed, train_seed=train_seed, n_tasks=n_tasks)
    done = 0

    @torch.inference_mode()
    def generate_fn(messages: list[dict[str, str]]) -> str:
        nonlocal done
        model.eval()
        text = generate_from_messages(
            model,
            tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
        )
        done += 1
        print(f"  {tag} {done}/{n_tasks}", flush=True)
        return text

    report = run_heldout(cfg, generate_fn=generate_fn)
    metrics = metrics_to_dict(report.metrics)
    gates = {
        "protocol": metrics["protocol_valid_rate"] >= GATE_PROTOCOL,
        "success": metrics["success_rate"] >= GATE_SUCCESS,
        "unknown": metrics["unknown_primitive_rate"] <= GATE_UNKNOWN,
    }
    print(tag, metrics, "gates", gates, flush=True)
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"metrics": metrics, "gates": gates}


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args.out.mkdir(parents=True, exist_ok=True)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
        print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    print("building dataset", {"n": args.train_n, "seed": args.train_seed})
    examples = build_dataset(
        SftDatasetConfig(seed=args.train_seed, n_tasks=args.train_n, include_query_prefix=True)
    )

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"loading tokenizer {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("loading base weights (cpu)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    if torch.cuda.is_available():
        log("moving model to cuda...")
        model = model.to("cuda")
        log(f"on cuda alloc_gb={round(torch.cuda.memory_allocated() / 1024**3, 2)}")
    log("enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    log("wrapping LoRA...")
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(LORA_TARGET_MODULES),
        ),
    )
    model.print_trainable_parameters()
    if torch.cuda.is_available():
        log(f"lora ready alloc_gb={round(torch.cuda.memory_allocated() / 1024**3, 2)}")

    if not args.skip_baseline:
        print("=== baseline ===")
        baseline = evaluate(
            tag="baseline",
            model=model,
            tokenizer=tokenizer,
            eval_seed=args.eval_seed,
            train_seed=args.train_seed,
            n_tasks=min(args.baseline_n, args.eval_n),
            max_new_tokens=args.max_new_tokens,
        )
        _write_json(args.out / "baseline_heldout.json", baseline)

    rows = []
    for example in examples:
        messages = [
            {
                "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                "content": m.content,
            }
            for m in example.messages
        ]
        rows.append(tokenize_sft_example(tokenizer, messages, max_seq_length=args.seq))
    train_ds = Dataset.from_list(rows)
    print("train rows", len(train_ds))

    train_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=max(1, args.max_steps // 20),
        logging_steps=5,
        save_steps=max(50, args.max_steps // 2),
        save_total_limit=2,
        max_steps=args.max_steps,
        optim="adamw_torch",
        seed=3407,
        report_to="none",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=PrefixLmCollator(pad_token_id=tokenizer.pad_token_id),
    )

    print(
        "=== train ===",
        {
            "max_steps": args.max_steps,
            "seq": args.seq,
            "lora_r": args.lora_r,
            "grad_accum": args.grad_accum,
            "train_n": args.train_n,
        },
        flush=True,
    )
    result = trainer.train()
    print(result)

    final_dir = args.out / "final_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print("saved", final_dir)

    if not args.skip_final:
        print("=== final ===")
        final = evaluate(
            tag="final",
            model=model,
            tokenizer=tokenizer,
            eval_seed=args.eval_seed,
            train_seed=args.train_seed,
            n_tasks=args.eval_n,
            max_new_tokens=args.max_new_tokens,
        )
        _write_json(args.out / "final_heldout.json", final)

    if torch.cuda.is_available():
        print("VRAM peak allocated GB", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
