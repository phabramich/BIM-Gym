from __future__ import annotations

from typing import Any

import torch

from bim_gym.hrm.constants import HrmConditionKind
from bim_gym.hrm.prompt import messages_to_instruction


@torch.inference_mode()
def generate_from_messages(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    do_sample: bool = True,
    condition: HrmConditionKind = HrmConditionKind.Reasoning,
) -> str:
    """Генерация с PrefixLM mask (token_type_ids=1 на весь prompt)."""
    prompt = messages_to_instruction(messages, condition=condition)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])

    eos_id = tokenizer.eos_token_id
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": eos_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    gen = out[0][prompt_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()
