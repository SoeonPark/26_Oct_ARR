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

if __package__:
    from .compare_runs import read_alignment_loss
else:
    from compare_runs import read_alignment_loss


# eval_massive_in_ko_loss  ->  ("massive", "in", "ko")
# eval_align_out_de-en_loss -> ("align", "out", "de-en")
METRIC_PATTERN = re.compile(r"^eval_(massive|wmt25|wmt23|align)_(in|out)_(.+)_loss$")

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
        default=None,
        help=(
            "Group to minimise, as <task>_<scope> (massive_in, wmt25_in, wmt23_in, "
            "align_in) or <task>_all. Defaults to the saved downstream task's "
            "in-language validation, or align_in for contrastive_only. "
            "Selecting on out-language data breaks the fully-unseen claim. Use "
            "final_step for a predetermined final checkpoint, especially "
            "gap-only runs where minimum variance can also reflect collapse."
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


def completed_final_step(run_dir, state_path, state):
    """Require the declared budget to be completed before final-step selection."""
    num_steps = None
    for path in (
        state_path.parent / "experiment_config.json",
        run_dir / "experiment_config.json",
        run_dir / "run_metadata.json",
    ):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if path.name == "run_metadata.json":
            config = config.get("experiment_config", {})
        if "num_steps" in config:
            num_steps = config["num_steps"]
            break
    if type(num_steps) is not int or num_steps <= 0:
        raise SystemExit(
            "final_step requires a positive num_steps in saved experiment_config.json "
            "or run_metadata.json. Restore the original run configuration to verify "
            "its predetermined training budget."
        )
    actual_step = state.get("global_step")
    if actual_step != num_steps:
        raise SystemExit(
            f"final_step requires the configured {num_steps} steps to be completed; "
            f"trainer_state.json records {actual_step}. Resume the run to its "
            "declared budget before selecting its final checkpoint."
        )
    return num_steps


def root_adapter_matches_step(run_dir, step):
    config_path = run_dir / "experiment_config.json"
    if not config_path.is_file():
        return False
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    return config.get("checkpoint_global_step") == step and any(
        (run_dir / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def load_run_config(run_dir, state_path):
    for path in (run_dir / "experiment_config.json",
                 state_path.parent / "experiment_config.json",
                 run_dir / "run_metadata.json"):
        if path.is_file():
            config = json.loads(path.read_text(encoding="utf-8"))
            return config.get("experiment_config", {}) if path.name == "run_metadata.json" else config
    return {}


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()

    state, state_path = load_trainer_state(run_dir)
    run_config = load_run_config(run_dir, state_path)
    downstream_task = run_config.get("downstream_task", "massive")
    if args.rule is None:
        args.rule = ("align_in" if run_config.get("training_type") == "contrastive_only"
                     else f"{downstream_task}_in")
    alignment_loss = read_alignment_loss(run_dir, state_path)
    steps = collect_losses(state.get("log_history", []))

    if not steps and args.rule != "final_step":
        raise SystemExit(
            f"No eval_*_loss entries in {state_path}. Was the run trained with "
            "eval_strategy=steps?"
        )

    print(f"Run   : {run_dir}")
    print(f"State : {state_path}")
    print(f"Alignment loss: {alignment_loss}")
    if args.rule == "final_step":
        print("Rule  : final_step (no validation-based checkpoint selection)")
    else:
        print(f"Rule  : {args.rule} (lower is better, language macro average)")
    if alignment_loss == "gap_consistency" and args.rule.startswith("align_"):
        print(
            "Gap variance alone can favor collapsed representations. "
            "Use a predetermined final_step for gap-only experiments, "
            "or massive_in when selecting for task transfer."
        )

    # Out-language data is reserved for the final evaluation. Selecting on it
    # leaks fr/de/it supervision into the pipeline through model choice, even
    # though no out-language gradient was ever taken.
    if args.rule.startswith(("massive_out", "wmt25_out", "wmt23_out", "align_out")) or args.rule.endswith("_all"):
        print(
            "\nWARNING: selecting on an out-language group uses unseen-language "
            "validation data for model selection. That conflicts with a "
            f"fully-unseen transfer claim. Prefer {downstream_task}_in (task methods) or "
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

    if args.rule == "final_step":
        scored = [(completed_final_step(run_dir, state_path, state), None)]
    else:
        scored = [
            (step, score_for_rule(groups, args.rule))
            for step, groups in steps.items()
        ]
        scored = [(step, score) for step, score in scored if score is not None]
        # eval_on_start and different save/eval intervals produce validation
        # rows without saved adapters. Only a loadable model is a candidate.
        unavailable = [step for step, _ in scored
                       if not (run_dir / f"checkpoint-{step}").is_dir()
                       and not root_adapter_matches_step(run_dir, step)]
        if unavailable:
            print(f"Excluded unsaved validation steps: {unavailable}")
            scored = [(step, score) for step, score in scored if step not in unavailable]

    if not scored:
        raise SystemExit(
            f"Rule '{args.rule}' has no saved checkpoint with matching validation. Available groups: "
            f"{group_names}"
        )

    best_step, best_score = (
        scored[0] if args.rule == "final_step"
        else min(scored, key=lambda pair: pair[1])
    )

    print(f"Best step  : {best_step}")
    if best_score is not None:
        print(f"Best score : {best_score:.4f}")

    # A 100k-step method is evaluated twice as often as a 50k one, so taking the
    # minimum over all of its steps gives it more chances to win. Report the
    # count so the comparison can be equalised, or fall back to final-step.
    print(f"Candidates : {len(scored)}")

    # A step is only usable if save_steps produced a checkpoint for it. This is
    # why eval_steps should be a multiple of save_steps.
    checkpoint_dir = run_dir / f"checkpoint-{best_step}"
    if (
        not checkpoint_dir.is_dir()
        and root_adapter_matches_step(run_dir, best_step)
    ):
        checkpoint_dir = run_dir
    if checkpoint_dir.is_dir():
        config_path = checkpoint_dir / "experiment_config.json"
        if config_path.is_file():
            downstream_task = json.loads(config_path.read_text()).get("downstream_task", "massive")
        scope = "out" if downstream_task == "wmt25" else "both"
        print(f"Checkpoint : {checkpoint_dir}")
        print(f"\nEvaluate it with:\n"
              f"  python3 evaluate.py --checkpoint_path {checkpoint_dir} \\\n"
              f"    --split test --language_scope {scope} --tasks alignment {downstream_task}")
    else:
        available = sorted(
            int(path.name.split("-")[-1])
            for path in run_dir.glob("checkpoint-*")
        )
        print(f"Checkpoint : MISSING ({checkpoint_dir})")
        if args.rule == "final_step":
            print(
                "The configured final step has no matching checkpoint. A root "
                "adapter is usable only when experiment_config.json records "
                f"checkpoint_global_step={best_step}. Saved steps: {available}"
            )
        else:
            print(
                "\nThe best validation step has no checkpoint. Set eval_steps to a "
                "multiple of save_steps so every evaluated step is saved.\n"
                f"Saved steps: {available}"
            )


if __name__ == "__main__":
    main()
