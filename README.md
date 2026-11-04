# BIM-Gym

RL gym for spatial reasoning on BIM models.

Generates synthetic building scenes and spatial-reasoning tasks, evaluates
model answers, and builds SFT datasets — with scripts for training small
reasoning models (HRM-style) locally or in Colab.

## Install

```bash
pip install -e ".[dev]"        # core + tests
pip install -e ".[dev,train]"  # + torch/transformers/peft for training
```

## Run

```bash
pytest                          # test suite
python -m bim_gym.sft --help    # dataset generation
```

See `scripts/` for training entry points (Colab notebook, local HRM, SFT smoke).
