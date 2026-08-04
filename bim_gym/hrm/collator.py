from __future__ import annotations

from typing import Any

import torch

from bim_gym.hrm.constants import HrmConditionKind
from bim_gym.hrm.prompt import split_sft_messages


def tokenize_sft_example(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_seq_length: int,
    condition: HrmConditionKind = HrmConditionKind.Reasoning,
) -> dict[str, list[int]]:
    """Токенизирует instruction/response с усечением под max_seq_length."""
    instruction, response = split_sft_messages(messages, condition=condition)
    instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
        response_ids = [*response_ids, eos_id]

    max_len = max(8, int(max_seq_length))
    if len(instruction_ids) + len(response_ids) > max_len:
        keep_instr = min(len(instruction_ids), max(1, max_len // 2))
        instruction_ids = instruction_ids[:keep_instr]
        keep_resp = max_len - len(instruction_ids)
        response_ids = response_ids[:keep_resp]
        if not response_ids and eos_id is not None:
            response_ids = [eos_id]

    return {
        "instruction_ids": instruction_ids,
        "response_ids": response_ids,
    }


class PrefixLmCollator:
    """PrefixLM collator из документации transformers HRM-Text."""

    def __init__(self, pad_token_id: int, ignore_label_id: int = -100):
        self.pad_token_id = int(pad_token_id)
        self.ignore_label_id = int(ignore_label_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_ids = [list(b["instruction_ids"]) + list(b["response_ids"]) for b in batch]
        prefix_lens = [len(b["instruction_ids"]) for b in batch]
        max_len = max(len(ids) for ids in full_ids)

        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, self.ignore_label_id)
        attention_mask = torch.zeros_like(input_ids)

        for i, (ids, plen) in enumerate(zip(full_ids, prefix_lens, strict=True)):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            token_type_ids[i, :plen] = 1
            labels[i, plen:length] = input_ids[i, plen:length]
            attention_mask[i, :length] = 1

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
