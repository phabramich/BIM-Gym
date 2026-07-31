from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bim_gym.sft.models import SftDatasetConfig, SftExample, example_to_jsonable


def export_jsonl(examples: list[SftExample], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(example_to_jsonable(example), ensure_ascii=False) for example in examples]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export bootstrap SFT dataset as JSONL")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--n", type=int, default=50, help="Number of tasks to generate")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--no-query-prefix",
        action="store_true",
        help="Disable query-prefix augmentation",
    )
    parser.add_argument(
        "--error-recovery-rate",
        type=float,
        default=0.0,
        help="Fraction of examples with intentional recovery steps",
    )
    args = parser.parse_args(argv)

    from bim_gym.sft.builder import build_dataset

    config = SftDatasetConfig(
        seed=args.seed,
        n_tasks=args.n,
        include_query_prefix=not args.no_query_prefix,
        error_recovery_rate=args.error_recovery_rate,
    )
    examples = build_dataset(config)
    export_jsonl(examples, args.out)
    print(f"Wrote {len(examples)} examples to {args.out}")


if __name__ == "__main__":
    main()
