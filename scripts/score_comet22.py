"""Score saved WMT predictions with the reference-based Mid-Align COMET-22 model."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comet22_scoring import score_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prediction_dir", type=Path,
                        help="One scope directory, e.g. CHECKPOINT/evaluations/test/in. No recursive search.")
    inputs.add_argument("--predictions", type=Path, nargs="+", help="Explicit prediction JSONL files from one scope.")
    parser.add_argument("--task", choices=("wmt23", "wmt25"), default="wmt23")
    parser.add_argument("--output_dir", type=Path, help="Defaults to prediction_dir (required with --predictions).")
    parser.add_argument("--checkpoint", help="Optional local copy of the pinned official checkpoints/model.ckpt, with ../hparams.yaml.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpus", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.prediction_dir is not None:
        args.prediction_dir = args.prediction_dir.expanduser()
        paths = sorted(args.prediction_dir.glob(f"{args.task}_predictions.*.jsonl"))
        output_dir = args.output_dir or args.prediction_dir
    else:
        paths, output_dir = args.predictions, args.output_dir
        if output_dir is None:
            parser.error("--output_dir is required with --predictions")
    report = score_files(paths, output_dir, args.task, checkpoint=args.checkpoint,
                         batch_size=args.batch_size, gpus=args.gpus)
    for direction, result in report["by_language"].items():
        print(f"{direction}: COMET-22={result['comet22_x100']:.4f} (x100; n={result['num_examples']})")
    print(f"Saved {args.task}_comet22_metrics.json and {args.task}_comet22_scores.jsonl to {output_dir}")


if __name__ == "__main__":
    main()
