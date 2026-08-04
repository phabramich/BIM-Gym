"""Стаб torch.distributed C API для ROCm Windows (USE_DISTRIBUTED=0).

Вызывать до import transformers / peft.
"""
from __future__ import annotations

import sys
import types
from datetime import timedelta
from pathlib import Path
from typing import Any


def _empty_type(name: str, bases: tuple[type, ...] = (object,)) -> type:
    return type(name, bases, {})


def install_rocm_distributed_stubs() -> bool:
    """Вернуть True, если стабы реально поставлены."""
    if getattr(install_rocm_distributed_stubs, "_done", False):
        return False

    try:
        import torch

        if torch.distributed.is_available():
            install_rocm_distributed_stubs._done = True  # type: ignore[attr-defined]
            return False
    except Exception:
        pass

    # 1) C-extension stub — нужен, чтобы import-цепочки не падали на ModuleNotFoundError
    c10d_name = "torch._C._distributed_c10d"
    if c10d_name not in sys.modules:
        c10d = types.ModuleType(c10d_name)
        store = _empty_type("Store")
        for attr, value in {
            "Store": store,
            "FileStore": _empty_type("FileStore", (store,)),
            "TCPStore": _empty_type("TCPStore", (store,)),
            "PrefixStore": _empty_type("PrefixStore", (store,)),
            "ProcessGroup": _empty_type("ProcessGroup"),
            "FakeProcessGroup": _empty_type("FakeProcessGroup"),
            "Work": _empty_type("Work"),
            "ReduceOp": _empty_type("ReduceOp"),
            "_DEFAULT_PG_TIMEOUT": timedelta(minutes=30),
            "_DEFAULT_PG_NCCL_TIMEOUT": timedelta(minutes=30),
        }.items():
            setattr(c10d, attr, value)

        def _getattr(attr: str) -> Any:
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            return _empty_type(attr)

        c10d.__getattr__ = _getattr  # type: ignore[attr-defined]
        sys.modules[c10d_name] = c10d

    # 2) Символы на torch.distributed до циклических импортов
    import torch.distributed as dist

    c10d = sys.modules[c10d_name]
    for attr in ("Store", "FileStore", "TCPStore", "PrefixStore", "ProcessGroup"):
        if not hasattr(dist, attr):
            setattr(dist, attr, getattr(c10d, attr))

    # 3) Dynamo тянет FSDP → FakeProcessGroup; подставляем лёгкие модули
    fake_pg = types.ModuleType("torch.testing._internal.distributed.fake_pg")
    fake_pg.FakeProcessGroup = c10d.FakeProcessGroup  # type: ignore[attr-defined]
    fake_pg.FakeStore = c10d.Store  # type: ignore[attr-defined]
    sys.modules.setdefault("torch.testing._internal.distributed.fake_pg", fake_pg)

    fully_shard = types.ModuleType("torch.distributed.fsdp._fully_shard")
    fully_shard._fsdp_param_group = None  # type: ignore[attr-defined]
    sys.modules.setdefault("torch.distributed.fsdp._fully_shard", fully_shard)

    flat = types.ModuleType("torch.distributed.fsdp._flat_param")
    flat.FlatParameter = _empty_type("FlatParameter")  # type: ignore[attr-defined]
    sys.modules.setdefault("torch.distributed.fsdp._flat_param", flat)

    fsdp_mod = types.ModuleType("torch.distributed.fsdp")
    fsdp_mod.FlatParameter = flat.FlatParameter  # type: ignore[attr-defined]
    fsdp_mod.FullyShardedDataParallel = _empty_type("FullyShardedDataParallel")  # type: ignore[attr-defined]
    fsdp_mod.CPUOffloadPolicy = _empty_type("CPUOffloadPolicy")  # type: ignore[attr-defined]
    fsdp_mod.MixedPrecisionPolicy = _empty_type("MixedPrecisionPolicy")  # type: ignore[attr-defined]
    fsdp_mod.OffloadPolicy = _empty_type("OffloadPolicy")  # type: ignore[attr-defined]
    fsdp_mod._fully_shard = fully_shard  # type: ignore[attr-defined]
    fsdp_mod._flat_param = flat  # type: ignore[attr-defined]
    sys.modules.setdefault("torch.distributed.fsdp", fsdp_mod)

    install_rocm_distributed_stubs._done = True  # type: ignore[attr-defined]
    return True


def patch_transformers_continuous_batching() -> Path | None:
    """Guard DeviceMesh import в установленном transformers (идемпотентно)."""
    try:
        import transformers
    except Exception:
        return None

    target = (
        Path(transformers.__file__).resolve().parent
        / "generation"
        / "continuous_batching"
        / "distributed.py"
    )
    if not target.exists():
        return None
    text = target.read_text(encoding="utf-8")
    marker = "# BIM-Gym ROCm guard"
    if marker in text:
        return target
    old = (
        "import torch\n"
        "import torch.distributed as dist\n"
        "from torch.distributed.tensor.device_mesh import DeviceMesh\n"
    )
    new = (
        "import torch\n"
        "import torch.distributed as dist\n"
        f"{marker}\n"
        "try:\n"
        "    from torch.distributed.tensor.device_mesh import DeviceMesh\n"
        "except Exception:  # ROCm Windows / USE_DISTRIBUTED=0\n"
        "    DeviceMesh = object  # type: ignore[misc, assignment]\n"
    )
    if old not in text:
        return None
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return target
