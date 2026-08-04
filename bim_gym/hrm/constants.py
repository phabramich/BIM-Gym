from __future__ import annotations

from enum import Enum


class HrmConditionKind(str, Enum):
    """Префикс условия из претрейна HRM-Text (см. model card)."""

    Reasoning = "reasoning"
    Plain = "plain"


DEFAULT_MODEL_ID = "sapientinc/HRM-Text-1B"
MIN_TRANSFORMERS_VERSION = "5.9.0"

# synth,cot composite — reasoning / CoT style (Sapient model card)
CONDITION_REASONING = "<|quad_end|><|object_ref_end|>"
CONDITION_PLAIN = ""

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# После конвертации чекпоинта HF: gqkv → gate/q/k/v; MLP стандартный
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def condition_token(kind: HrmConditionKind = HrmConditionKind.Reasoning) -> str:
    if kind is HrmConditionKind.Reasoning:
        return CONDITION_REASONING
    return CONDITION_PLAIN
