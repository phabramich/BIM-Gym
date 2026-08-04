"""Локальный smoke: HRM-Text-1B + PrefixLM LoRA (без Unsloth / 4-bit)."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.pop("BNB_ROCM_VERSION", None)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ROCm Windows torch часто без distributed — до transformers/peft
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

from bim_gym.hrm import (
    DEFAULT_MODEL_ID,
    LORA_TARGET_MODULES,
    PrefixLmCollator,
    tokenize_sft_example,
)
from bim_gym.sft import SftDatasetConfig, build_dataset

OUT = Path(r"c:\dev\BIM-Gym\out\local_sft_smoke")
OUT.mkdir(parents=True, exist_ok=True)

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

model_name = os.environ.get("BIM_GYM_MODEL", DEFAULT_MODEL_ID)
max_seq_length = int(os.environ.get("BIM_GYM_SEQ", "768"))
max_steps = int(os.environ.get("BIM_GYM_MAX_STEPS", "4"))

print("building tiny dataset...")
examples = build_dataset(SftDatasetConfig(seed=0, n_tasks=24, include_query_prefix=True))

print("loading", model_name, "fp16 LoRA...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# device_map='auto' на ROCm Windows часто падает в safetensors (0xC0000005)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    attn_implementation="sdpa",
)
if torch.cuda.is_available():
    model = model.to("cuda")
model.gradient_checkpointing_enable()
model = get_peft_model(
    model,
    LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    ),
)
model.print_trainable_parameters()
if torch.cuda.is_available():
    print("alloc_gb", round(torch.cuda.memory_allocated() / 1024**3, 2))

rows = []
for example in examples:
    messages = [
        {
            "role": m.role.value if hasattr(m.role, "value") else str(m.role),
            "content": m.content,
        }
        for m in example.messages
    ]
    rows.append(tokenize_sft_example(tokenizer, messages, max_seq_length=max_seq_length))

train_ds = Dataset.from_list(rows)
print("train rows", len(train_ds))

args = TrainingArguments(
    output_dir=str(OUT),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=1,
    logging_steps=1,
    save_steps=50,
    save_total_limit=1,
    max_steps=max_steps,
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
    args=args,
    train_dataset=train_ds,
    data_collator=PrefixLmCollator(pad_token_id=tokenizer.pad_token_id),
)

print("training...")
result = trainer.train()
print(result)

final_dir = OUT / "final_lora"
model.save_pretrained(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print("saved", final_dir)
if torch.cuda.is_available():
    print("VRAM peak allocated GB", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
