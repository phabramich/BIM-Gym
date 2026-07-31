# %% [markdown]
# BIM-Gym — SFT bootstrap (Colab)
#
# Файл с ячейками `# %%` — удобно конвертировать в `.ipynb`.
# Цель фазы 0: протокол DSL + базовые цепочки на synthetic tasks.
#
# Перед запуском:
# 1. Runtime → GPU (T4 ок).
# 2. Укажи `REPO_URL` ниже (или залей репо в `/content/BIM-Gym`).
# 3. Drive-бэкап не нужен для ~1–2ч; чекпоинты живут в `/content` до disconnect.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Данные / репо
    repo_dir: Path = Path("/content/BIM-Gym")
    # Обязательно подставь свой URL, иначе clone не сработает
    repo_url: str = "https://github.com/<you>/BIM-Gym.git"
    # Если задан существующий JSONL — не генерируем train на лету
    train_jsonl: Path | None = None

    train_n: int = 2000
    train_seed: int = 0
    eval_n: int = 64
    eval_seed: int = 10_000  # != train_seed

    # Модель (T4: Qwen3.5-4B bf16 LoRA ~10GB VRAM).
    # Unsloth: QLoRA 4-bit для Qwen3.5 НЕ рекомендуется.
    # Альтернатива: unsloth/gemma-4-E4B-it (нужен HF token)
    # Не для free Colab: Qwen3.6 (27B+)
    model_name: str = "unsloth/Qwen3.5-4B"
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    load_in_16bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Train
    output_dir: Path = Path("/content/bim_gym_sft_out")
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    # Mid-train held-out дорогой (generate×N). Для первого прогона — 0; смотри baseline+final.
    eval_every_steps: int = 0
    eval_max_new_tokens: int = 1024
    seed: int = 3407

    # Гейты фазы 0 (из спеки)
    gate_protocol_valid: float = 0.90
    gate_success: float = 0.55
    gate_unknown_primitive: float = 0.02


CFG = Config()

# %% [markdown]
# ## Вводные и ожидаемые результаты
#
# ### Вводные
# | Вводная | Зачем |
# |---------|--------|
# | Маленькая instruct модель (Qwen3.5-4B / Gemma4-E4B) | Субъект обучения |
# | SFT JSONL из `bim_gym.sft` | system/user/assistant, assistant = actions JSON |
# | Held-out задачи с **другим seed** | Честный protocol/success |
# | Runtime `bim_gym` (предикаты) | Успех без LLM-judge |
# | GPU ~16GB (T4) | bf16 LoRA 4B (~10GB) |
#
# ### Что реально учим на этом этапе
# Скриптовые reference-траектории → **грамматика протокола** и шаблоны
# `query→assert` / `move→assert`. Это bootstrap, не проверка гипотезы
# «модель сама находит нетривиальные пространственные стратегии» (это RL).
#
# ### Метрики и гейты
# | Метрика | Смысл | Гейт |
# |---------|--------|------|
# | `protocol_valid_rate` | JSON + известные kind/op | ≥ 0.90 |
# | `unknown_primitive_rate` | мусорные примитивы | < 0.02 |
# | `success_rate` | `run_task` по goals | ≥ 0.55 |
# | train loss | CE | стабильно ↓ |
#
# ### Интерпретация
# - protocol ↑, success низкий → формат ок, якоря/действия слабые
# - оба низкие → мало данных / формат слишком жёсткий / модель мала
# - гейты на held-out пройдены → можно в RL фазы 1; иначе крутим SFT, не RL
#
# ### Чего не ждать от первого Colab-прогона
# Экстраполяция на длинные цепочки, Revit transfer, победа над frontier zero-shot.

# %%
import json
import subprocess
import sys


def sh(cmd: str) -> None:
    print("+", cmd)
    subprocess.check_call(cmd, shell=True)


if not CFG.repo_dir.exists():
    if "<you>" in CFG.repo_url:
        raise RuntimeError(
            "Задай CFG.repo_url на свой GitHub/GitLab репозиторий BIM-Gym "
            "или залей код в /content/BIM-Gym вручную."
        )
    sh(f'git clone --depth 1 "{CFG.repo_url}" "{CFG.repo_dir}"')

sh(f'pip install -q -e "{CFG.repo_dir}"')
sh("pip install -q datasets trl peft accelerate bitsandbytes")
# Unsloth тянет нужный transformers (для Qwen3.5 — v5)
sh('pip install -q -U "unsloth" "unsloth_zoo"')

# %%
import torch
from datasets import Dataset
from transformers import TrainerCallback
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(CFG.repo_dir))

from bim_gym.eval import (  # noqa: E402
    HeldoutEvalConfig,
    metrics_to_dict,
    run_heldout,
)
from bim_gym.sft import SftDatasetConfig, build_dataset  # noqa: E402
from bim_gym.sft.export import export_jsonl, load_jsonl  # noqa: E402

# %%
# Датасет


def to_hf_dataset(rows: list[dict]) -> Dataset:
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
        print(f"Loaded train JSONL: {len(rows)} ← {CFG.train_jsonl}")
        return to_hf_dataset(rows)

    print(f"Generating train n={CFG.train_n} seed={CFG.train_seed}")
    examples = build_dataset(
        SftDatasetConfig(seed=CFG.train_seed, n_tasks=CFG.train_n, include_query_prefix=True)
    )
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    out = CFG.output_dir / "sft_train.jsonl"
    export_jsonl(examples, out)
    print(f"Wrote {out}")
    return to_hf_dataset([e.model_dump(mode="json") for e in examples])


train_ds = load_or_build_train()
heldout_cfg = HeldoutEvalConfig(seed=CFG.eval_seed, train_seed=CFG.train_seed, n_tasks=CFG.eval_n)
print("train:", len(train_ds), "heldout_n:", CFG.eval_n, "heldout_seed:", CFG.eval_seed)
print(
    "steps/epoch ≈",
    len(train_ds)
    // max(1, CFG.per_device_train_batch_size * CFG.gradient_accumulation_steps),
)

# %%
# Модель + LoRA

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CFG.model_name,
    max_seq_length=CFG.max_seq_length,
    load_in_4bit=CFG.load_in_4bit,
    load_in_16bit=CFG.load_in_16bit,
    dtype=None,
    fast_inference=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=CFG.lora_r,
    lora_alpha=CFG.lora_alpha,
    lora_dropout=CFG.lora_dropout,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=CFG.seed,
    max_seq_length=CFG.max_seq_length,
)

# %%
# Генерация для held-out


@torch.inference_mode()
def generate_fn(messages: list[dict[str, str]]) -> str:
    FastLanguageModel.for_inference(model)
    template_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **template_kwargs,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=CFG.eval_max_new_tokens,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def evaluate_model(tag: str, n_tasks: int | None = None) -> dict:
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


class HeldoutCallback(TrainerCallback):
    def __init__(self, every_steps: int, out_dir: Path):
        self.every_steps = every_steps
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_steps <= 0:
            return
        if state.global_step == 0 or state.global_step % self.every_steps != 0:
            return
        row = {
            "step": state.global_step,
            **evaluate_model(f"step-{state.global_step}", n_tasks=min(32, CFG.eval_n)),
        }
        with (self.out_dir / "heldout_history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Вернуть модель в train-режим после for_inference
        model.train()


# %%
# Trainer

CFG.output_dir.mkdir(parents=True, exist_ok=True)

sft_kwargs = dict(
    output_dir=str(CFG.output_dir),
    per_device_train_batch_size=CFG.per_device_train_batch_size,
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,
    learning_rate=CFG.learning_rate,
    warmup_ratio=CFG.warmup_ratio,
    logging_steps=CFG.logging_steps,
    save_steps=CFG.save_steps,
    save_total_limit=CFG.save_total_limit,
    num_train_epochs=CFG.num_train_epochs,
    max_steps=CFG.max_steps,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    seed=CFG.seed,
    report_to="none",
    packing=False,
    dataset_num_proc=1,
)
try:
    sft_args = SFTConfig(**sft_kwargs, max_seq_length=CFG.max_seq_length)
except TypeError:
    sft_args = SFTConfig(**sft_kwargs)

callbacks = []
if CFG.eval_every_steps > 0:
    callbacks.append(HeldoutCallback(CFG.eval_every_steps, CFG.output_dir))

try:
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=sft_args,
        callbacks=callbacks,
    )
except TypeError:
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=sft_args,
        callbacks=callbacks,
    )

# %%
print("=== baseline ===")
baseline = evaluate_model("baseline", n_tasks=min(32, CFG.eval_n))
(CFG.output_dir / "baseline_heldout.json").write_text(
    json.dumps(baseline, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
model.train()

# %%
result = trainer.train()
print(result)

final_dir = CFG.output_dir / "final_lora"
model.save_pretrained(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print("saved", final_dir)

# %%
print("=== final ===")
final = evaluate_model("final", n_tasks=CFG.eval_n)
(CFG.output_dir / "final_heldout.json").write_text(
    json.dumps(final, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# %% [markdown]
# Чекпоинты: `output_dir/checkpoint-*`, адаптер: `final_lora/`.
# В конце скачай `final_lora/` и `final_heldout.json` локально.
# В RL идём только если gates.protocol / success / unknown закрыты на held-out.
