# %% [markdown]
# BIM-Gym — SFT bootstrap (Colab) on HRM-Text-1B
#
# Чекпоинты в `/content/bim_gym_sft_out`; копия последнего в `/content/bim_gym_sft_latest`.
# Drive не используется — скачивай вручную из Colab Files.
# PrefixLM + token_type_ids; без Unsloth / chat_template.

# %%
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass
class Config:
    repo_dir: Path = Path("/content/BIM-Gym")
    repo_url: str = "https://github.com/<you>/BIM-Gym.git"
    train_jsonl: Path | None = None

    train_n: int = 2000
    train_seed: int = 0
    eval_n: int = 64
    eval_seed: int = 10_000
    eval_mid_n: int = 16
    baseline_n: int = 32

    model_name: str = "sapientinc/HRM-Text-1B"
    max_seq_length: int = 2048
    load_in_4bit: bool = False

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    output_dir: Path = Path("/content/bim_gym_sft_out")
    latest_dir: Path = Path("/content/bim_gym_sft_latest")
    resume_from_checkpoint: bool = True

    num_train_epochs: float = 1.0
    max_steps: int = 100
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 40
    save_total_limit: int = 3
    eval_every_steps: int = 0
    eval_max_new_tokens: int = 1024
    seed: int = 3407

    plateau_min_step: int = 40
    plateau_patience: int = 3
    plateau_min_delta: float = 0.005

    gate_protocol_valid: float = 0.90
    gate_success: float = 0.55
    gate_unknown_primitive: float = 0.02


CFG = Config()


def latest_checkpoint(out_dir: Path) -> str | None:
    if not out_dir.exists():
        return None
    ckpts = sorted(
        (p for p in out_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


def publish_latest(src: Path, latest_dir: Path | None = None) -> None:
    target = Path(latest_dir or CFG.latest_dir)
    target.mkdir(parents=True, exist_ok=True)
    dst = target / src.name
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    (target / "LAST.txt").write_text(str(dst), encoding="utf-8")
    print("published for download:", dst)


def setup_output_dir(cfg: Config) -> Config:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.latest_dir.mkdir(parents=True, exist_ok=True)
    print("output_dir:", cfg.output_dir)
    print("download_dir:", cfg.latest_dir)
    return cfg


CFG = setup_output_dir(CFG)

# %%
def sh(cmd: str) -> None:
    print("+", cmd)
    subprocess.check_call(cmd, shell=True)


if not CFG.repo_dir.exists():
    if "<you>" in CFG.repo_url:
        raise RuntimeError(
            "Задай CFG.repo_url или залей код в /content/BIM-Gym (zip-ячейка сверху)."
        )
    sh(f'git clone --depth 1 "{CFG.repo_url}" "{CFG.repo_dir}"')

sh(f'pip install -q -e "{CFG.repo_dir}"')
# HRM-Text нужен нативный hrm_text в transformers
sh('pip install -q "transformers>=5.9.0" datasets peft accelerate')
try:
    sh("pip install -q bitsandbytes")
except Exception as exc:
    print("bitsandbytes skip:", exc)

# %%
import gc

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore[misc, assignment]

sys.path.insert(0, str(CFG.repo_dir))

from bim_gym.eval import (  # noqa: E402
    HeldoutEvalConfig,
    metrics_to_dict,
    run_heldout,
)
from bim_gym.hrm import (  # noqa: E402
    LORA_TARGET_MODULES,
    PrefixLmCollator,
    generate_from_messages,
    tokenize_sft_example,
)
from bim_gym.sft import SftDatasetConfig, build_dataset  # noqa: E402
from bim_gym.sft.export import export_jsonl, load_jsonl  # noqa: E402

USE_BF16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "bf16", USE_BF16)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

# %%
def to_message_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "id": row.get("id", ""),
                "messages": [
                    {
                        "role": m["role"] if isinstance(m["role"], str) else str(m["role"]),
                        "content": m["content"],
                    }
                    for m in row["messages"]
                ],
            }
            for row in rows
        ]
    )


def load_or_build_train() -> Dataset:
    if CFG.train_jsonl and Path(CFG.train_jsonl).exists():
        rows = load_jsonl(CFG.train_jsonl)
        print(f"Loaded train JSONL: {len(rows)} <- {CFG.train_jsonl}")
        return to_message_dataset(rows)

    print(f"Generating train n={CFG.train_n} seed={CFG.train_seed}")
    examples = build_dataset(
        SftDatasetConfig(seed=CFG.train_seed, n_tasks=CFG.train_n, include_query_prefix=True)
    )
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    out = CFG.output_dir / "sft_train.jsonl"
    export_jsonl(examples, out)
    print(f"Wrote {out}")
    publish_latest(out)
    return to_message_dataset([e.model_dump(mode="json") for e in examples])


train_ds = load_or_build_train()
print("train:", len(train_ds), "max_steps:", CFG.max_steps)

# %%
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_kwargs: dict = {
    "attn_implementation": "sdpa",
    "device_map": "auto",
}
if CFG.load_in_4bit:
    if BitsAndBytesConfig is None:
        raise RuntimeError("load_in_4bit requires bitsandbytes + BitsAndBytesConfig")
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
elif USE_BF16:
    model_kwargs["dtype"] = torch.bfloat16
else:
    model_kwargs["dtype"] = torch.float16
    print("no bf16 → fp16 LoRA (HRM-1B normally fits T4 without 4-bit)")

model = AutoModelForCausalLM.from_pretrained(CFG.model_name, **model_kwargs)
model.gradient_checkpointing_enable()
model = get_peft_model(
    model,
    LoraConfig(
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    ),
)
model.print_trainable_parameters()

# %%
def restore_train_mode() -> None:
    model.train()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@torch.inference_mode()
def generate_fn(messages: list[dict[str, str]]) -> str:
    model.eval()
    return generate_from_messages(
        model,
        tokenizer,
        messages,
        max_new_tokens=CFG.eval_max_new_tokens,
        temperature=0.2,
        do_sample=True,
    )


def evaluate_model(tag: str, n_tasks: int | None = None) -> dict:
    try:
        cfg = HeldoutEvalConfig(
            seed=CFG.eval_seed,
            train_seed=CFG.train_seed,
            n_tasks=n_tasks or CFG.eval_n,
        )
        report = run_heldout(cfg, generate_fn=generate_fn)
        metrics = metrics_to_dict(report.metrics)
        gates = {
            "protocol": metrics["protocol_valid_rate"] >= CFG.gate_protocol_valid,
            "success": metrics["success_rate"] >= CFG.gate_success,
            "unknown": metrics["unknown_primitive_rate"] <= CFG.gate_unknown_primitive,
        }
        print(tag, metrics, "gates", gates)
        return {"metrics": metrics, "gates": gates}
    finally:
        restore_train_mode()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    publish_latest(path)


class PublishLatestCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt.exists():
            publish_latest(ckpt)


class LossPlateauEarlyStopCallback(TrainerCallback):
    def __init__(self, min_step: int, patience: int, min_delta: float):
        self.min_step = min_step
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.stale = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        loss = float(logs["loss"])
        step = state.global_step
        if loss < self.best - self.min_delta:
            self.best = loss
            self.stale = 0
            return
        if step < self.min_step:
            return
        self.stale += 1
        print(
            f"loss plateau {self.stale}/{self.patience} "
            f"(loss={loss:.4f}, best={self.best:.4f})"
        )
        if self.stale >= self.patience:
            print(f"early-stop at step {step}: train loss plateau")
            control.should_training_stop = True
            control.should_save = True


class HeldoutCallback(TrainerCallback):
    def __init__(self, every_steps: int, out_dir: Path, n_tasks: int):
        self.every_steps = every_steps
        self.out_dir = Path(out_dir)
        self.n_tasks = n_tasks
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_steps <= 0:
            return
        if state.global_step == 0 or state.global_step % self.every_steps != 0:
            return
        row = {
            "step": state.global_step,
            **evaluate_model(f"step-{state.global_step}", n_tasks=self.n_tasks),
        }
        hist = self.out_dir / "heldout_history.jsonl"
        with hist.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        publish_latest(hist)
        restore_train_mode()


# %%
CFG.output_dir.mkdir(parents=True, exist_ok=True)


def _tokenize_batch(examples: dict) -> dict:
    instruction_ids: list[list[int]] = []
    response_ids: list[list[int]] = []
    for messages in examples["messages"]:
        row = tokenize_sft_example(
            tokenizer,
            messages,
            max_seq_length=CFG.max_seq_length,
        )
        instruction_ids.append(row["instruction_ids"])
        response_ids.append(row["response_ids"])
    return {
        "instruction_ids": instruction_ids,
        "response_ids": response_ids,
    }


train_tok = train_ds.map(_tokenize_batch, batched=True, remove_columns=train_ds.column_names)
print("tokenized:", len(train_tok))

eff_batch = CFG.per_device_train_batch_size * CFG.gradient_accumulation_steps
steps_per_epoch = max(1, len(train_tok) // eff_batch)
planned = CFG.max_steps if CFG.max_steps > 0 else int(steps_per_epoch * CFG.num_train_epochs)
warmup_steps = max(1, int(planned * CFG.warmup_ratio))

optim_name = "adamw_torch"
try:
    import bitsandbytes  # noqa: F401

    optim_name = "adamw_8bit"
except Exception:
    pass

train_args = TrainingArguments(
    output_dir=str(CFG.output_dir),
    per_device_train_batch_size=CFG.per_device_train_batch_size,
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,
    learning_rate=CFG.learning_rate,
    warmup_steps=warmup_steps,
    logging_steps=CFG.logging_steps,
    save_steps=CFG.save_steps,
    save_total_limit=CFG.save_total_limit,
    num_train_epochs=CFG.num_train_epochs,
    max_steps=CFG.max_steps,
    lr_scheduler_type="cosine",
    optim=optim_name,
    seed=CFG.seed,
    report_to="none",
    bf16=USE_BF16,
    fp16=not USE_BF16,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

callbacks = [
    PublishLatestCallback(),
    LossPlateauEarlyStopCallback(
        min_step=CFG.plateau_min_step,
        patience=CFG.plateau_patience,
        min_delta=CFG.plateau_min_delta,
    ),
]
if CFG.eval_every_steps > 0:
    callbacks.append(
        HeldoutCallback(CFG.eval_every_steps, CFG.output_dir, n_tasks=CFG.eval_mid_n)
    )

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_tok,
    data_collator=PrefixLmCollator(pad_token_id=tokenizer.pad_token_id),
    callbacks=callbacks,
)

# %%
print("=== baseline ===")
if CFG.resume_from_checkpoint and latest_checkpoint(CFG.output_dir):
    print("skip baseline (resuming from checkpoint)")
else:
    baseline = evaluate_model("baseline", n_tasks=min(CFG.baseline_n, CFG.eval_n))
    _write_json(CFG.output_dir / "baseline_heldout.json", baseline)
restore_train_mode()

# %%
resume_path = latest_checkpoint(CFG.output_dir) if CFG.resume_from_checkpoint else None
if resume_path:
    print("resuming from", resume_path)
result = trainer.train(resume_from_checkpoint=resume_path)
print(result)

final_dir = CFG.output_dir / "final_lora"
model.save_pretrained(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
publish_latest(final_dir)
print("saved", final_dir)

# %%
print("=== final ===")
final = evaluate_model("final", n_tasks=CFG.eval_n)
_write_json(CFG.output_dir / "final_heldout.json", final)
print("download:", CFG.latest_dir)
if torch.cuda.is_available():
    print("VRAM peak allocated GB", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
