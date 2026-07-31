from __future__ import annotations

from bim_gym.eval.models import EvalScore
from bim_gym.eval.parse import detect_unknown_primitive, parse_assistant_actions
from bim_gym.tasks.runner import run_task
from bim_gym.tasks.spec import TaskSpec


def score_generation(task: TaskSpec, assistant_text: str) -> EvalScore:
    unknown_primitive = detect_unknown_primitive(assistant_text)
    actions = parse_assistant_actions(assistant_text)
    if actions is None:
        return EvalScore(
            protocol_valid=False,
            unknown_primitive=unknown_primitive,
            success=False,
            error="invalid assistant protocol",
        )

    result = run_task(task, actions)
    return EvalScore(
        protocol_valid=True,
        unknown_primitive=False,
        success=result.success,
        error=result.error,
    )
