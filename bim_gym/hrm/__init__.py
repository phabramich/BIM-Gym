from bim_gym.hrm.collator import PrefixLmCollator, tokenize_sft_example
from bim_gym.hrm.constants import (
    DEFAULT_MODEL_ID,
    LORA_TARGET_MODULES,
    MIN_TRANSFORMERS_VERSION,
    HrmConditionKind,
)
from bim_gym.hrm.generate import generate_from_messages
from bim_gym.hrm.prompt import (
    messages_to_instruction,
    messages_to_response,
    split_sft_messages,
)
from bim_gym.hrm.rocm_compat import (
    install_rocm_distributed_stubs,
    patch_transformers_continuous_batching,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "LORA_TARGET_MODULES",
    "MIN_TRANSFORMERS_VERSION",
    "HrmConditionKind",
    "PrefixLmCollator",
    "generate_from_messages",
    "install_rocm_distributed_stubs",
    "messages_to_instruction",
    "messages_to_response",
    "patch_transformers_continuous_batching",
    "split_sft_messages",
    "tokenize_sft_example",
]
