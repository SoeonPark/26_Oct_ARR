"""Pick a checkpoint from validation losses recorded during training.

`main.py` deliberately leaves `metric_for_best_model` unset: a macro over
languages is not one of the keys Trainer emits, and `_determine_best_metric`
raises on a missing key. Every per-language `eval_*_loss` still lands in
`trainer_state.json`, so selection happens here instead — which also means the
selection rule can change without retraining.

Usage:
    python3 scripts/select_checkpoint.py RUN_DIR [--rule massive_out] [--table]
"""

import argparse
import json
from pathlib import Path
import re


# eval_massive_in_ko_loss  ->  ("massive", "in", "ko")
# eval_align_out_de-en_loss -> ("align", "out", "de-en")
METRIC_PATTERN = re.compile(r"^eval_(massive|align)_(in|out)_(.+)_loss$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select a checkpoint from recorded validation losses."
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Run directory containing trainer_state.json.",
    )
    parser.add_argument(
        "--rule",
        type=str,
        default="massive_in",
        help=(
            "Group to minimise, as <task>_<scope> (massive_in, align_in, "
            "massive_out, align_out) or <task>_all. Defaults to massive_in: "
            "selecting on an out-language group would use fr/de/it "
            "labels or parallel data for model selection and break the "
            "fully-unseen claim. Use align_in for contrastive_only, which "
            "never trains on the task."
        ),
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print every evaluated step instead of only the selection.",
    )
    return parser.parse_args()


def load_trainer_state(run_dir):
    """Read trainer_state.json from the run root or its newest checkpoint."""
    state_path = run_dir / "trainer_state.json"

    if not state_path.is_file():
        checkpoints = sorted(
            run_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"No trainer_state.json and no checkpoints under {run_dir}."
            )
        state_path = checkpoints[-1] / "trainer_state.json"

    with state_path.open("r", encoding="utf-8") as state_file:
        return json.load(state_file), state_path


def collect_losses(log_history):
    """Merge the per-dataset log rows into {step: {group: {lang: loss}}}."""
    steps = {}

    for entry in log_history:
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


def macro(group_losses):
    """Language macro average, as required by the evaluation protocol."""
    if not group_losses:
        return None
    return sum(group_losses.values()) / len(group_losses)


def score_for_rule(step_groups, rule):
    if rule.endswith("_all"):
        task = rule[: -len("_all")]
        languages = {}
        for scope in ("in", "out"):
            languages.update(step_groups.get(f"{task}_{scope}", {}))
        return macro(languages)

    return macro(step_groups.get(rule, {}))


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()

    state, state_path = load_trainer_state(run_dir)
    steps = collect_losses(state.get("log_history", []))

    if not steps:
        raise SystemExit(
            f"No eval_*_loss entries in {state_path}. Was the run trained with "
            "eval_strategy=steps?"
        )

    print(f"Run   : {run_dir}")
    print(f"State : {state_path}")
    print(f"Rule  : {args.rule} (lower is better, language macro average)")

    # Out-language data is reserved for the final evaluation. Selecting on it
    # leaks fr/de/it supervision into the pipeline through model choice, even
    # though no out-language gradient was ever taken.
    if args.rule.startswith(("massive_out", "align_out")):
        print(
            "\nWARNING: selecting on an out-language group uses fr/de/it "
            "validation data for model selection. That conflicts with a "
            "fully-unseen transfer claim. Prefer massive_in (task methods) or "
            "align_in (contrastive_only)."
        )

    print()

    group_names = sorted({name for groups in steps.values() for name in groups})

    if args.table:
        header = f"{'step':>8}  " + "  ".join(f"{n:>16}" for n in group_names)
        print(header)
        print("-" * len(header))
        for step in sorted(steps):
            cells = []
            for name in group_names:
                value = macro(steps[step].get(name, {}))
                cells.append("---".rjust(16) if value is None else f"{value:16.4f}")
            print(f"{step:>8}  " + "  ".join(cells))
        print()

    scored = [
        (step, score_for_rule(groups, args.rule))
        for step, groups in steps.items()
    ]
    scored = [(step, score) for step, score in scored if score is not None]

    if not scored:
        raise SystemExit(
            f"Rule '{args.rule}' matched nothing. Available groups: "
            f"{group_names}"
        )

    best_step, best_score = min(scored, key=lambda pair: pair[1])

    print(f"Best step  : {best_step}")
    print(f"Best score : {best_score:.4f}")

    # A 100k-step method is evaluated twice as often as a 50k one, so taking the
    # minimum over all of its steps gives it more chances to win. Report the
    # count so the comparison can be equalised, or fall back to final-step.
    print(f"Candidates : {len(scored)} evaluated steps")

    # A step is only usable if save_steps produced a checkpoint for it. This is
    # why eval_steps should be a multiple of save_steps.
    checkpoint_dir = run_dir / f"checkpoint-{best_step}"
    if checkpoint_dir.is_dir():
        print(f"Checkpoint : {checkpoint_dir}")
        print(f"\nEvaluate it with:\n"
              f"  python3 evaluator.py --checkpoint_path {checkpoint_dir} \\\n"
              f"    --split test --language_scope both --tasks alignment massive")
    else:
        available = sorted(
            int(path.name.split("-")[-1])
            for path in run_dir.glob("checkpoint-*")
        )
        print(f"Checkpoint : MISSING ({checkpoint_dir})")
        print(
            "\nThe best validation step has no checkpoint. Set eval_steps to a "
            "multiple of save_steps so every evaluated step is saved.\n"
            f"Saved steps: {available}"
        )


if __name__ == "__main__":
    main()
