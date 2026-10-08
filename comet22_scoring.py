"""Reference-based COMET-22 scoring, without importing MT training dependencies.

Mid-Align passes sources/predictions/references to HF evaluate's COMET wrapper.
For unbabel-comet >= 2 that wrapper uses Unbabel/wmt22-comet-da. We call the
same model directly, pin its weights, and retain the original segment order.
"""

from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import tempfile


MODEL_ID = "Unbabel/wmt22-comet-da"
MODEL_REVISION = "2760a223ac957f30acfb18c8aa649b01cf1d75f2"
CHECKPOINT_SHA256 = "e213091cde220f97b89f8bdfa750c458cfea741ad62affb455b59900210ff2af"
HPARAMS_SHA256 = "265ef22345ea5b9ffa020a7fe5be613a95ff931c44bf8d09a26d96c6c6048f60"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comet_inputs(rows):
    if not rows:
        raise ValueError("COMET-22 requires at least one prediction.")
    samples = []
    for index, row in enumerate(rows):
        for key in ("source", "prediction", "target"):
            value = row.get(key)
            if not isinstance(value, str):
                raise ValueError(f"Row {index}: {key} must be a string.")
            # Empty hypotheses are model failures to score, not rows to discard.
            if key != "prediction" and not value.strip():
                raise ValueError(f"Row {index}: {key} must not be empty.")
        samples.append({"src": row["source"], "mt": row["prediction"], "ref": row["target"]})
    return samples


def checked_scores(scores, count):
    scores = [float(score) for score in scores]
    if len(scores) != count or not all(math.isfinite(score) for score in scores):
        raise ValueError("COMET-22 returned missing, extra, or non-finite segment scores.")
    return scores


def direction_of(row):
    direction = row.get("direction") or row.get("lang")
    if not isinstance(direction, str) or not direction:
        raise ValueError("Each prediction needs a direction or lang.")
    # Legacy WMT records use lang=de to mean en-de.
    return direction if "-" in direction else f"en-{direction}"


def summarize_scores(rows, scores, metadata):
    scores = checked_scores(scores, len(rows))
    if not rows:
        raise ValueError("Cannot summarize an empty prediction set.")
    grouped = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[direction_of(row)].append(score)
    by_language = {}
    for direction, values in sorted(grouped.items()):
        mean = math.fsum(values) / len(values)
        by_language[direction] = {
            "direction": direction, "num_examples": len(values),
            "comet22": mean, "comet22_x100": mean * 100,
        }

    def macro(directions):
        if not directions:
            return None
        mean = math.fsum(by_language[key]["comet22"] for key in directions) / len(directions)
        return {"num_directions": len(directions), "comet22": mean, "comet22_x100": mean * 100}

    mean = math.fsum(scores) / len(scores)
    return {
        "schema_version": 1, "status": "scored", "metric": "comet22",
        "num_examples": len(rows),
        # HF evaluate's mean_score is the segment mean, not the language macro.
        "mean_score": mean, "mean_score_x100": mean * 100,
        "by_language": by_language,
        "macro_average": macro(list(by_language)),
        "macro_by_translation_direction": {
            "en_to_x": macro([key for key in by_language if key.startswith("en-")]),
            "x_to_en": macro([key for key in by_language if key.endswith("-en")]),
        },
        "scorer_metadata": metadata,
    }


class Comet22Scorer:
    def __init__(self, checkpoint=None, batch_size=16, gpus=0):
        if batch_size <= 0 or gpus not in (0, 1):
            raise ValueError("COMET-22 requires batch_size > 0 and gpus=0 or 1.")
        try:
            import comet
            import torch
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "Install requirements-comet22.txt in a separate environment; "
                "run scripts/score_comet22.py with that Python."
            ) from error
        if gpus and not torch.cuda.is_available():
            raise RuntimeError("COMET-22 requested a GPU but CUDA is unavailable; use --gpus 0 for CPU.")
        if checkpoint is None:
            snapshot = Path(snapshot_download(
                repo_id=MODEL_ID, revision=MODEL_REVISION,
                allow_patterns=["checkpoints/model.ckpt", "hparams.yaml"],
            ))
            checkpoint = snapshot / "checkpoints" / "model.ckpt"
        # Keep the snapshot symlink layout: COMET expects ../hparams.yaml.
        checkpoint = Path(checkpoint).expanduser().absolute()
        hparams = checkpoint.parent.parent / "hparams.yaml"
        for path, expected in ((checkpoint, CHECKPOINT_SHA256), (hparams, HPARAMS_SHA256)):
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"Not the pinned {MODEL_ID} artifact (SHA256 mismatch): {path}")
        self.model = comet.load_from_checkpoint(str(checkpoint))
        self.model.eval()
        self.batch_size, self.gpus = batch_size, gpus
        self.metadata = {
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256,
            "hparams_sha256": HPARAMS_SHA256,
            "reference_based": True, "batch_size": batch_size, "gpus": gpus,
            "precision": "32-true", "mc_dropout": 0, "text_preprocessing": "none",
            "score_scale": "raw model output; *_x100 fields multiply it by 100",
            "versions": {name: version(name) for name in (
                "unbabel-comet", "torch", "transformers", "pytorch-lightning",
                "numpy", "huggingface-hub", "tokenizers", "sentencepiece",
            )},
        }

    def score(self, rows):
        output = self.model.predict(
            comet_inputs(rows), batch_size=self.batch_size, gpus=self.gpus,
            accelerator="gpu" if self.gpus else "cpu",
            mc_dropout=0, progress_bar=False, num_workers=0,
        )
        # COMET restores segment order after its internal length batching.
        scores = checked_scores(output.scores, len(rows))
        if not math.isfinite(float(output.system_score)):
            raise ValueError("COMET-22 returned a non-finite system score.")
        return scores, {**self.metadata, "system_score": float(output.system_score)}


def score_comet22(predictions, args):
    """Bridge used by evaluate.py; COMET never imports into the training process."""
    comet_inputs(predictions)
    python = getattr(args, "comet22_python", None)
    if not python:
        raise ValueError("Set --comet22_python to the Python with requirements-comet22.txt installed.")
    with tempfile.TemporaryDirectory(prefix="octarr-comet22-") as temporary:
        root = Path(temporary)
        source = root / "predictions.jsonl"
        write_jsonl(source, predictions)
        command = [
            str(python), str(Path(__file__).parent / "scripts" / "score_comet22.py"),
            "--predictions", str(source), "--output_dir", str(root),
            "--batch_size", str(getattr(args, "comet22_batch_size", 16)),
            "--gpus", str(getattr(args, "comet22_gpus", 0)),
        ]
        if getattr(args, "comet22_checkpoint", None):
            command.extend(["--checkpoint", str(args.comet22_checkpoint)])
        subprocess.run(command, check=True)
        report = json.loads((root / "wmt23_comet22_metrics.json").read_text(encoding="utf-8"))
        scored_rows = read_jsonl(root / "wmt23_comet22_scores.jsonl")
        expected = [(direction_of(row), row["sample_id"]) for row in predictions]
        actual = [(direction_of(row), row["sample_id"]) for row in scored_rows]
        if actual != expected:
            raise ValueError("COMET-22 subprocess changed the prediction row order or identities.")
        return checked_scores([row["comet22"] for row in scored_rows], len(predictions)), report["scorer_metadata"]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def score_files(paths, output_dir, task="wmt23", *, checkpoint=None, batch_size=16, gpus=0):
    paths = [Path(path).expanduser().absolute() for path in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide nonempty, distinct prediction files from one evaluation scope.")
    rows, provenance, seen = [], [], set()
    for path in paths:
        records = read_jsonl(path)
        comet_inputs(records)
        for row in records:
            key = (direction_of(row), row.get("sample_id"))
            if not isinstance(key[1], str) or not key[1] or key in seen:
                raise ValueError(f"Missing or duplicate sample_id for {key[0]} in {path}.")
            seen.add(key)
        rows.extend(records)
        provenance.append({"path": str(path), "sha256": file_sha256(path), "num_examples": len(records)})
    scorer = Comet22Scorer(checkpoint=checkpoint, batch_size=batch_size, gpus=gpus)
    scores, metadata = scorer.score(rows)
    report = summarize_scores(rows, scores, metadata)
    report["prediction_files"] = provenance
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Separate artifacts preserve the existing BLEU/chrF evaluation report.
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".comet22-") as temporary:
        temporary = Path(temporary)
        score_path = temporary / f"{task}_comet22_scores.jsonl"
        metrics_path = temporary / f"{task}_comet22_metrics.json"
        write_jsonl(score_path, ({**row, "comet22": score, "comet22_x100": score * 100}
                                for row, score in zip(rows, scores)))
        metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        score_path.replace(output_dir / score_path.name)
        metrics_path.replace(output_dir / metrics_path.name)
    return report
