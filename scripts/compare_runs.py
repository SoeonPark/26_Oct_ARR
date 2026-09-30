"""Tabulate validation losses across runs in a directory.

Written for the diagnostic sweeps: scripts/lr_sweep.sh varies the learning rate
and scripts/layer_probe.sh varies the representation layer, and both are read by
comparing per-language validation losses at matching steps.

Usage:
    python3 scripts/compare_runs.py results_lr_sweep
    python3 scripts/compare_runs.py results_layer_probe --step 0
"""

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import resolve_alignment_loss


METRIC_PATTERN = re.compile(r"^eval_(massive|align)_(in|out)_(.+)_loss$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare validation losses across runs."
    )
    parser.add_argument(
        "root",
        type=str,
        help="Directory containing run directories (searched recursively).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Only report this step. Defaults to every evaluated step.",
    )
    return parser.parse_args()


def find_states(root):
    """One trainer_state.json per run, preferring the run root."""
    states = {}

    for path in sorted(root.rglob("trainer_state.json")):
        run_dir = path.parent
        if run_dir.name.startswith("checkpoint-"):
            run_dir = run_dir.parent
        # A run root state is written at the end and supersedes a checkpoint's.
        if run_dir in states and states[run_dir].parent == run_dir:
            continue
        states[run_dir] = path

    return states


def collect(state_path):
    with state_path.open("r", encoding="utf-8") as state_file:
        history = json.load(state_file).get("log_history", [])

    steps = {}
    for entry in history:
        step = entry.get("step")
        if step is None:
            continue
        for key, value in entry.items():
            match = METRIC_PATTERN.match(key)
            if match is None:
                continue
            task, scope, language = match.groups()
            group = steps.setdefault(step, {})
            group.setdefault(f"{task}_{scope}", {})[language] = value

    return steps


def macro(values):
    return sum(values.values()) / len(values) if values else None


def read_alignment_loss(run_dir, state_path):
    """Read the saved objective; older runs implicitly used InfoNCE."""
    candidates = [
        state_path.parent / "experiment_config.json",
        run_dir / "experiment_config.json",
        run_dir / "run_metadata.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if path.name == "run_metadata.json":
            config = config["experiment_config"]
        return resolve_alignment_loss(config)
    return "infonce"


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    states = find_states(root)
    if not states:
        raise SystemExit(f"No trainer_state.json found under {root}")

    rows = []
    group_names = set()

    for run_dir, state_path in states.items():
        alignment_loss = read_alignment_loss(run_dir, state_path)
        for step, groups in collect(state_path).items():
            if args.step is not None and step != args.step:
                continue
            scores = {
                name: macro(languages)
                for name, languages in groups.items()
            }
            group_names.update(scores)
            rows.append((run_dir.name, step, alignment_loss, scores))

    if not rows:
        at_step = "" if args.step is None else f" at step {args.step}"
        raise SystemExit(f"No eval_*_loss entries{at_step}")

    group_names = sorted(group_names)
    rows.sort(key=lambda row: (row[0], row[1]))

    name_width = max(len(row[0]) for row in rows)
    name_width = min(name_width, 78)

    header = f"{'run':<{name_width}}  {'step':>7}  {'alignment_loss':<15}  " + "  ".join(
        f"{name:>14}" for name in group_names
    )
    print(header)
    print("-" * len(header))

    for name, step, alignment_loss, scores in rows:
        cells = []
        for group in group_names:
            value = scores.get(group)
            cells.append("---".rjust(14) if value is None else f"{value:14.4f}")
        print(
            f"{name[:name_width]:<{name_width}}  {step:>7}  "
            f"{alignment_loss:<15}  " + "  ".join(cells)
        )

    print(
        "\nalign_* reports the selected alignment loss. Compare values only "
        "within the same loss type and evaluation batching. InfoNCE uses "
        "in-batch negatives; gap_consistency is within-batch distance variance. "
        "A small gap variance alone does not establish semantic alignment."
    )
    print(
        "massive_* on a contrastive_only run is a zero-shot diagnostic, not "
        "transfer: no task gradient was ever taken."
    )


if __name__ == "__main__":
    main()
