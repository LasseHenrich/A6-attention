"""
Generates submission.json for A6-attention.

Usage:
    uv run python submit.py a b c ...   # specific tasks
    uv run python submit.py             # all tasks

The resulting submission.json can be uploaded to the course webpage.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "grader"))

from grader import _RUNNERS, run_puzzles, write_json  # noqa: E402

_ALL_TASKS = list(_RUNNERS.keys())

if __name__ == "__main__":
    keys = sys.argv[1:] if len(sys.argv) > 1 else _ALL_TASKS
    bad = [k for k in keys if k not in _RUNNERS]
    if bad:
        print(f"Unknown task(s): {bad}. Valid tasks: {_ALL_TASKS}")
        sys.exit(1)

    data = run_puzzles(keys)
    out = _ROOT / "submission.json"
    write_json(data, out)
    print(f"Wrote {out}")
