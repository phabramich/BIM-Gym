from __future__ import annotations

from bim_gym.hrm.constants import (
    IM_END,
    IM_START,
    HrmConditionKind,
    condition_token,
)


def messages_to_instruction(
    messages: list[dict[str, str]],
    *,
    condition: HrmConditionKind = HrmConditionKind.Reasoning,
) -> str:
    """Схлопывает system+user в PrefixLM-инструкцию HRM (без chat_template)."""
    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            continue
        else:
            raise ValueError(f"unsupported message role: {role!r}")

    body_parts = [*system_parts, *user_parts]
    body = "\n\n".join(part for part in body_parts if part)
    return f"{IM_START}{condition_token(condition)}{body}{IM_END}"


def messages_to_response(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "assistant":
            return message["content"]
    raise ValueError("messages must include an assistant response")


def split_sft_messages(
    messages: list[dict[str, str]],
    *,
    condition: HrmConditionKind = HrmConditionKind.Reasoning,
) -> tuple[str, str]:
    return messages_to_instruction(messages, condition=condition), messages_to_response(messages)
