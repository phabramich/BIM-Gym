from __future__ import annotations

from bim_gym.hrm.constants import (
    CONDITION_REASONING,
    IM_END,
    IM_START,
    HrmConditionKind,
)
from bim_gym.hrm.prompt import (
    messages_to_instruction,
    messages_to_response,
    split_sft_messages,
)
from bim_gym.hrm.collator import PrefixLmCollator, tokenize_sft_example


class _FakeTok:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # один «токен» на символ — достаточно для unit-тестов коллатора
        return [ord(ch) % 97 + 10 for ch in text]


def test_messages_to_instruction_uses_prefixlm_markers() -> None:
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": '{"format":"bim_gym_actions_v1","actions":[]}'},
    ]
    instruction = messages_to_instruction(messages, condition=HrmConditionKind.Reasoning)
    assert instruction.startswith(IM_START + CONDITION_REASONING)
    assert instruction.endswith(IM_END)
    assert "SYS" in instruction and "USER" in instruction
    assert "bim_gym_actions_v1" not in instruction


def test_split_sft_messages() -> None:
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    instruction, response = split_sft_messages(messages)
    assert response == "A"
    assert messages_to_response(messages) == "A"
    assert instruction == messages_to_instruction(messages)


def test_prefixlm_collator_masks_loss_and_prefix() -> None:
    collator = PrefixLmCollator(pad_token_id=0, ignore_label_id=-100)
    batch = [
        {"instruction_ids": [1, 2, 3], "response_ids": [4, 5]},
        {"instruction_ids": [9], "response_ids": [8]},
    ]
    out = collator(batch)
    assert out["input_ids"].shape == (2, 5)
    assert out["token_type_ids"][0].tolist() == [1, 1, 1, 0, 0]
    assert out["labels"][0].tolist() == [-100, -100, -100, 4, 5]
    assert out["attention_mask"][0].tolist() == [1, 1, 1, 1, 1]
    assert out["token_type_ids"][1].tolist() == [1, 0, 0, 0, 0]
    assert out["labels"][1].tolist() == [-100, 8, -100, -100, -100]


def test_tokenize_sft_example_appends_eos_and_truncates() -> None:
    tok = _FakeTok()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    row = tokenize_sft_example(tok, messages, max_seq_length=12)
    assert row["response_ids"][-1] == tok.eos_token_id
    assert len(row["instruction_ids"]) + len(row["response_ids"]) <= 12
