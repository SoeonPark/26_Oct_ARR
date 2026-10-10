"""Run ALMA/OPUS Gap variants through training, retrieval, BLEU and COMET.

Uses the same final-adapter and evaluation checks as the existing experiment
queue. Each GPU finishes its own evaluation immediately after training.
"""

import argparse
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import parse_args as parse_training_args, validate_gap_config
from scripts.mt_retrieval_multiseed_queue import atomic_json
from scripts.run_massive_gap_variants import (
    MODELS, check_saved_plan, preflight as training_preflight, selected_losses,
)
from scripts.run_remote_experiments import MANIFEST_SHA256, RemoteQueue
from scripts.wmt23_pipeline import config_matches, file_sha256, find_completed, read_json


def build_manifest(args):
    data_dir = args.data_dir.expanduser().resolve()
    data_path = data_dir / "manifest.json"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Prepared ALMA/OPUS data not found: {data_path}\n"
            "Git does not include the dataset. Copy the entire wmt23_alma_ja_opus "
            "directory with rsync -aL, or pass --data-dir /path/to/prepared-data."
        )
    if file_sha256(data_path) != args.manifest_sha256:
        raise ValueError("MT manifest hash mismatch. Copy the original prepared data, "
                         "or supply the verified --manifest-sha256.")
    data = read_json(data_path)
    if data["corpus_profile"] != "alma_ja_opus" or data["data_seed"] != 42:
        raise ValueError("These experiments require alma_ja_opus with data seed 42.")
    model = MODELS[args.model]
    results_root = ROOT / "results_wmt23"
    jobs = []
    for variant in selected_losses(args):
        loss = f"gap_distance_{variant}"
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("ALIGNMENT_", "WMT23_", "TRAIN_", "TRAINING_", "WANDB_"))
               and key not in ("MODEL_NAME", "OUTPUT_ROOT", "DOWNSTREAM_MICRO_BATCH_SIZE")}
        env.update(DRY_RUN="1", PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES=str(args.gpu),
                   MODEL_NAME=model, ALIGNMENT_LOSS=loss, ALIGNMENT_BATCHING="same_pair",
                   TRAINING_SEED="42", TRAIN_EVAL_LANGUAGE_SCOPE="in",
                   TRAIN_SAMPLE_LOG_INTERVAL="1000", TRAIN_SAMPLE_LOG_LIMIT="8",
                   WMT23_DATA_DIR=str(data_dir), WMT23_MANIFEST_SHA256=args.manifest_sha256,
                   WMT23_CORPUS_PROFILE="alma_ja_opus", WMT23_DOWNSTREAM_SAMPLING="balanced_mixed",
                   WANDB_MODE=args.wandb_mode, OUTPUT_ROOT=str(results_root))
        script = "scripts/wmt23_contrastive_only.sh"
        preview = subprocess.run(["bash", script, "alternative"], cwd=ROOT, env=env,
                                 capture_output=True, text=True, check=True)
        lines = [line for line in preview.stdout.splitlines() if line.startswith("CUDA_VISIBLE_DEVICES=")]
        if len(lines) != 1:
            raise ValueError("Expected one WMT23 training command from the launcher.")
        command = shlex.split(lines[0])
        with patch.object(sys, "argv", ["main.py", *command[command.index("main.py") + 1:]]):
            config_args = parse_training_args()
        validate_gap_config(config_args)
        config = vars(config_args)
        expected = dict(model_name=model, downstream_task="wmt23", alignment_loss=loss,
                        training_type="alternative", alignment_hidden_state_layer=-1,
                        alignment_hidden_state_position="last_token", training_seed=42,
                        alignment_batching="same_pair", batch_size=16, accumulative_steps=1,
                        num_steps=100000, learning_rate=1e-4, warmup_ratio=.1,
                        peft_lora_r=16, peft_lora_alpha=32, peft_lora_dropout=.1,
                        quantization_compute_dtype="bfloat16", quantization_load_in_4bit=True,
                        quantization_use_double_quant=True, quantization_type="nf4",
                        alignment_temperature=.05, alignment_gap_scale=1.,
                        eval_language_scope="in", eval_steps=2500, eval_batch_size=16,
                        save_steps=1000, logging_steps=10, eval_sample_log_limit=64,
                        train_sample_log_interval=1000, train_sample_log_limit=8,
                        downstream_micro_batch_size=8 if args.model == "qwen" else 0,
                        training_anchor_langs="en", training_lang=["de", "cs", "ja"],
                        out_inference_lang=["zh", "ru", "uk"],
                        wmt23_data_dir=str(data_dir), wmt23_manifest_sha256=args.manifest_sha256,
                        wmt23_corpus_profile="alma_ja_opus", wmt23_downstream_sampling="balanced_mixed")
        if any(config[key] != value for key, value in expected.items()):
            raise ValueError(f"Launcher settings changed for {loss}: expected {expected}")
        config.pop("wandb_run_name")  # Launcher timestamps must not alter queue identity.
        jobs.append(dict(id=f"wmt23_{args.model}_alternative_{loss}_layerlast_seed42",
                         model=model, mode="alternative", loss=loss, gpu=args.gpu, steps=100000,
                         script=script, config=config, results_root=str(results_root),
                         evaluation_language_scopes=["in"]))
    pairs = ("de-en", "cs-en", "en-ja")
    return dict(
        schema_version=1, python=sys.executable,
        # Do not resolve the venv interpreter symlink: that loses the COMET environment.
        comet_python=str(args.comet_python.expanduser().absolute()),
        results_root=str(results_root), data_dir=str(data_dir), manifest_sha256=args.manifest_sha256,
        wandb_mode=args.wandb_mode, jobs=jobs, adopted={},
        evaluation_id="wmt23_gap_variants_final", evaluation_language_scopes=["in"],
        max_new_tokens=16384, wmt23_batch_size=16,
        wmt_eos_policy="generation_config_plus_tokenizer_v1",
        test_counts={"in": {key: data["counts"]["test"][key]
                            for lang in ("de", "cs", "ja") for key in (f"en-{lang}", f"{lang}-en")}},
        retrieval=dict(evaluation_id="wmt23_gap_variants_retrieval", language_scopes=["in"],
                       batch_size=16, chunk_size=256, pair_counts={pair: 2000 for pair in pairs},
                       pair_file_sha256={pair: data["files"][data["opus_files"][pair]["test"]]["sha256"]
                                         for pair in pairs}),
    )


def preflight(manifest, gpu):
    """Validate the prepared snapshot and both environments before training."""
    from data_utils import WMT23Dataset
    WMT23Dataset.validate_prepared_manifest(argparse.Namespace(**manifest["jobs"][0]["config"]))
    training_preflight(manifest, gpu)
    subprocess.run([manifest["python"], "-c", "import sacrebleu, pyarrow; from sacrebleu.metrics import BLEU; "
                    "BLEU(tokenize='ja-mecab').corpus_score(['テスト'], [['テスト']])"], check=True)
    subprocess.run([manifest["comet_python"], "-c",
                    "import comet, torch; "
                    "assert torch.cuda.is_available(), 'COMET environment requires CUDA'; "
                    "print('COMET environment: OK')"],
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, check=True)


class WMT23GapQueue(RemoteQueue):
    def train_job(self, gpu, job):
        if find_completed(job["results_root"], job) is None:
            model_root = Path(job["results_root"]) / job["model"].replace("/", "__")
            for path in model_root.glob("*/run_metadata.json"):
                config = read_json(path).get("experiment_config", {})
                if config_matches(job["config"], config):
                    raise RuntimeError(f"An incomplete matching run exists: {path.parent}. "
                                       "Inspect it before restarting; optimizer state is not saved.")
            saved = self.state["jobs"].get(job["id"], {})
            if saved.get("training_status") in {"running", "failed", "completed"}:
                raise RuntimeError(f"Previous training has no verified final adapter: {job['id']}. "
                                   "Inspect its state and logs before restarting.")
        super().train_job(gpu, job)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--loss", choices=("detach", "rms", "both"),
                        help="Default: llama runs detach then rms; qwen runs rms.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/wmt23_alma_ja_opus")
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--comet-python", type=Path, default=ROOT / ".venv-comet22/bin/python")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Print the plan; no GPU checks, files, or training.")
    modes.add_argument("--check-only", action="store_true", help="Check data, CUDA and both environments; no training.")
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be a nonnegative physical GPU index")
    if args.state_dir is None:
        loss_tag = "_".join(selected_losses(args))
        args.state_dir = ROOT / "logs" / f"wmt23_gap_{args.model}_{loss_tag}_gpu{args.gpu}"
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        manifest = build_manifest(args)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for index, job in enumerate(manifest["jobs"], 1):
        print(f"{index}. {job['model']} / ALMA+JA / alternative / {job['loss']} "
              "/ layer -1 / BF16 / seed 42 / 100000 steps", flush=True)
        print("   -> final IN OPUS retrieval -> WMT23 BLEU -> COMET-22 (GPU), then the next run", flush=True)
    print(f"State directory: {args.state_dir}", flush=True)
    if args.dry_run:
        return
    state_dir = args.state_dir.expanduser().resolve()
    check_saved_plan(state_dir, manifest)
    preflight(manifest, args.gpu)
    if args.check_only:
        print("Preflight passed. No training started.", flush=True)
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    with (state_dir / "controller.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_saved_plan(state_dir, manifest)
        atomic_json(state_dir / "manifest.json", manifest)
        WMT23GapQueue(manifest, state_dir).run()


if __name__ == "__main__":
    main()
