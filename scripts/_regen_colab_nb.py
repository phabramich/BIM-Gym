"""Собрать colab_sft_bootstrap.ipynb из .py (# %% ячейки)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "scripts" / "colab_sft_bootstrap.py"
OUT = ROOT / "scripts" / "colab_sft_bootstrap.ipynb"

ZIP_CELL = '''from pathlib import Path
import shutil
import zipfile

ZIP = Path("/content/BIM-Gym-master.zip")
REPO = Path("/content/BIM-Gym")

if not ZIP.exists():
    raise FileNotFoundError(
        "Залей BIM-Gym-master.zip в /content (Colab Files → Upload) и перезапусти ячейку."
    )

with zipfile.ZipFile(ZIP, "r") as zf:
    zf.extractall("/content")

extracted = Path("/content/BIM-Gym-master")
if not extracted.is_dir():
    raise FileNotFoundError(
        "В архиве нет папки BIM-Gym-master — проверь, что это GitHub Download ZIP репозитория."
    )

if REPO.exists():
    shutil.rmtree(REPO)
extracted.rename(REPO)
print("ok →", REPO)
'''

INTRO_MD = """# BIM-Gym — SFT bootstrap (Colab) on **HRM-Text-1B**

Фаза 0: DSL-протокол + базовые action-цепочки на synthetic BIM tasks.

- Модель: `sapientinc/HRM-Text-1B` (PrefixLM + `token_type_ids`, без chat_template / Unsloth)
- Train: 2000 examples, LoRA r=16, seq 2048, 100 steps
- Gates: protocol ≥ 0.90, success ≥ 0.55, unknown ≤ 0.02
- Выход: `/content/bim_gym_sft_out` (+ копия в `bim_gym_sft_latest` для скачивания)

**Перед запуском**
1. Runtime → **GPU** (T4 хватает; L4/A100 быстрее).
2. GitHub → Download ZIP → залей как `BIM-Gym-master.zip` в `/content` → первая ячейка.
   Или задай `CFG.repo_url` на свой fork и пропусти zip.
3. Нужен `transformers>=5.9.0` (ставится в install-ячейке).

Ожидание на T4: train ~15–35 мин + held-out generation дольше.
"""


def _lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def main() -> None:
    raw = PY.read_text(encoding="utf-8")
    chunks = raw.split("# %%")
    cells: list[dict] = [
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "zip-upload",
            "metadata": {},
            "outputs": [],
            "source": _lines(ZIP_CELL),
        },
        {
            "cell_type": "markdown",
            "id": "intro-md",
            "metadata": {},
            "source": _lines(INTRO_MD),
        },
    ]

    for i, chunk in enumerate(chunks):
        body = chunk.lstrip()
        if not body.strip():
            continue
        if body.startswith("[markdown]"):
            md = body[len("[markdown]") :].lstrip("\n").rstrip() + "\n"
            # intro уже в INTRO_MD
            if md.startswith("# BIM-Gym — SFT bootstrap"):
                continue
            cells.append(
                {
                    "cell_type": "markdown",
                    "id": f"md-{i}",
                    "metadata": {},
                    "source": _lines(md),
                }
            )
            continue
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "id": f"code-{i}",
                "metadata": {},
                "outputs": [],
                "source": _lines(body.rstrip() + "\n"),
            }
        )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
