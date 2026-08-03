"""Run infrastructure: config hashing, run directories, artifact I/O.

All provided — students never modify this module.

The config is complete from Task A (see ``config.py``), so ``config_hash``
is **stable across the whole assignment** — adding nothing and editing
nothing keeps every run's identity fixed.  No device/hardware string ever
enters a hashed artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from attention.config import ExperimentConfig
from attention.utils import seed_everything

__all__ = [
    "all_runs",
    "config_hash",
    "init_run",
    "latest_run",
    "seed_everything",
    "write_json",
    "write_metrics",
    "write_scoreboard",
]


def _canonical(cfg: ExperimentConfig) -> str:
    """JSON of the non-default fields, sorted by key (for the run hash)."""
    defaults = {f.name: f.default for f in fields(cfg)}
    non_default = {f.name: getattr(cfg, f.name) for f in fields(cfg) if getattr(cfg, f.name) != defaults[f.name]}
    return json.dumps(non_default, sort_keys=True, separators=(",", ":"))


def config_hash(cfg: ExperimentConfig) -> str:
    """8-hex-digit SHA-256 of the canonical non-default-field representation.

    Two configs that differ only in fields still at their default value
    produce the same hash, so adding a field with a default never changes
    the identity of existing runs.
    """
    return hashlib.sha256(_canonical(cfg).encode()).hexdigest()[:8]


def init_run(
    cfg: ExperimentConfig,
    results_root: Path,
    tag: str | None = None,
) -> Path:
    """Create ``results_root/runs/YYYYMMDDThhmmss-<hash>[-<tag>]/``.

    Seeds the global RNGs from ``cfg.init_seed`` and writes ``config.yaml``.
    """
    seed_everything(cfg.init_seed)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_name = f"{timestamp}-{config_hash(cfg)}"
    if tag is not None:
        run_name = f"{run_name}-{tag}"
    run_dir = results_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    with open(run_dir / "config.yaml", "w") as fh:
        yaml.dump(config_dict, fh, default_flow_style=False, sort_keys=True)
    return run_dir


def latest_run(results_root: Path, tag: str) -> Path | None:
    """Newest ``results_root/runs/*-<tag>`` directory, or ``None``.

    Run names start with a second-resolution timestamp, so the
    lexicographic maximum is the most recent run for that tag.
    """
    candidates = all_runs(results_root, tag)
    return candidates[-1] if candidates else None


def all_runs(results_root: Path, tag: str) -> list[Path]:
    """Every ``results_root/runs/*-<tag>`` directory, oldest first.

    Figures that compare a model against an earlier version of itself (Task G's
    leaky-vs-correct mask grids) read every run for a tag rather than only the
    newest.
    """
    return sorted(d for d in (results_root / "runs").glob(f"*-{tag}") if d.is_dir())


def write_json(obj: Any, path: Path) -> None:
    """Write *obj* to *path* as indented JSON (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def write_metrics(run_dir: Path, history: dict[str, Any]) -> None:
    """Write a run's ``History`` dict to ``run_dir/metrics.json``."""
    write_json(history, run_dir / "metrics.json")


def write_scoreboard(run_dir: Path, scoreboard: dict[str, Any]) -> None:
    """Write a battery scoreboard to ``run_dir/scoreboard.json``."""
    write_json(scoreboard, run_dir / "scoreboard.json")
