"""Run portable MASSIVE Gap variants, evaluating each before the next.

No prepared MT data, COMET environment, or local queue/PID manifest is needed.
"""

import argparse
import fcntl
from importlib.metadata import version
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
from scripts.run_remote_experiments import RemoteQueue
from scripts.wmt23_pipeline import Pipeline, config_matches, find_completed, read_json

MODELS = {"llama": "meta-llama/Llama-3.2-1B-Instruct", "qwen": "Qwen/Qwen3.5-2B"}
TRAINING_PACKAGES = {
    "transformers", "torch", "accelerate", "datasets", "peft", "bitsandbytes",
    "wandb", "numpy", "huggingface-hub", "pyarrow",
}


def selected_losses(args):
    choice = args.loss or ("both" if args.model == "llama" else "rms")
    return ("detach", "rms") if choice == "both" else (choice,)


def build_manifest(args):
    model = MODELS[args.model]
    results_root = ROOT / "results"
    jobs = []
    for variant in selected_losses(args):
        loss = f"gap_distance_{variant}"
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("ALIGNMENT_", "WMT23_", "TRAIN_", "TRAINING_", "WANDB_"))
               and key not in ("MODEL_NAME", "OUTPUT_ROOT", "DOWNSTREAM_MICRO_BATCH_SIZE")}
        env.update(DRY_RUN="1", PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES=str(args.gpu),
                   MODEL_NAME=model, ALIGNMENT_LOSS=loss, ALIGNMENT_BATCHING="same_pair",
                   ALIGNMENT_HIDDEN_STATE_LAYER="-1", TRAINING_SEED="42", TRAIN_EVAL_LANGUAGE_SCOPE="in",
                   TRAIN_SAMPLE_LOG_INTERVAL="1000", TRAIN_SAMPLE_LOG_LIMIT="8",
                   WANDB_MODE=args.wandb_mode, OUTPUT_ROOT=str(results_root))
        script = "scripts/massive_contrastive_only.sh"
        preview = subprocess.run(["bash", script, "alternative"], cwd=ROOT, env=env,
                                 capture_output=True, text=True, check=True)
        lines = [line for line in preview.stdout.splitlines() if line.startswith("CUDA_VISIBLE_DEVICES=")]
        if len(lines) != 1:
            raise ValueError("Expected one MASSIVE training command from the launcher.")
        command = shlex.split(lines[0])
        with patch.object(sys, "argv", ["main.py", *command[command.index("main.py") + 1:]]):
            config_args = parse_training_args()
        validate_gap_config(config_args)
        config = vars(config_args)
        expected = dict(model_name=model, downstream_task="massive", alignment_loss=loss,
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
                        downstream_micro_batch_size=0, training_anchor_langs="en",
                        training_lang=["ko", "ja", "es"], out_inference_lang=["fr", "de", "it"])
        if any(config[key] != value for key, value in expected.items()):
            raise ValueError(f"Launcher settings changed for {loss}: expected {expected}")
        config.pop("wandb_run_name")  # The launcher timestamp must not alter queue identity.
        jobs.append(dict(id=f"massive_{args.model}_alternative_{loss}_layerlast_seed42",
                         model=model, mode="alternative", loss=loss, gpu=args.gpu, steps=100000,
                         lane=args.model, script=script, config=config, results_root=str(results_root)))
    return dict(
        schema_version=1, python=sys.executable, results_root=str(results_root),
        # Shared train_job forwards this unused environment value for all tasks.
        manifest_sha256="", wandb_mode=args.wandb_mode, jobs=jobs, adopted={},
        massive_post_training_evaluation=dict(
            enabled=True, evaluation_id="massive_gap_variants_final",
            language_scopes=["in", "out"], alignment_language_scopes=["in"],
            alignment_batch_size=16, massive_batch_size=16, retrieval_chunk_size=256,
            max_new_tokens=128, eval_sample_log_limit=64,
            save_alignment_embeddings=False, save_alignment_sample_metrics=True,
            pair_counts={"in": {pair: 2000 for pair in ("en-es", "en-ja", "en-ko")}},
            language_counts={"in": {lang: 2974 for lang in ("en", "ko", "ja", "es")},
                             "out": {lang: 2974 for lang in ("fr", "de", "it")}}),
    )


def preflight(manifest, gpu):
    """Check the training environment and selected model access, without weights."""
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        package, expected = line.split("==")
        package = package.split("[")[0]
        if package in TRAINING_PACKAGES and version(package).split("+")[0] != expected:
            raise RuntimeError(f"Install requirements.txt: {package}=={expected} is required.")
    code = (
        "import sys, torch, peft, bitsandbytes; "
        "from transformers import AutoConfig; "
        "assert torch.cuda.is_available(), 'Training requires CUDA'; "
        "assert torch.cuda.is_bf16_supported(), 'Training requires BF16 support'; "
        "AutoConfig.from_pretrained(sys.argv[1]); "
        "print('Training environment/model config access: OK')"
    )
    subprocess.run([sys.executable, "-c", code, manifest["jobs"][0]["model"]],
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, check=True)


class MassiveGapQueue(RemoteQueue):
    def environment(self, gpu):
        # Bypass RemoteQueue's MT-only data/environment requirements.
        env = Pipeline.environment(self, gpu)
        env["WANDB_MODE"] = self.manifest["wandb_mode"]
        return env

    def train_job(self, gpu, job):
        if find_completed(job["results_root"], job) is None:
            # main.py writes run_metadata before model loading, whereas its
            # root experiment_config may not appear until the final save.
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


def check_saved_plan(state_dir, manifest):
    path = state_dir / "manifest.json"
    state_path = state_dir / "state.json"
    if path.exists() and read_json(path) != manifest:
        raise RuntimeError("Saved queue settings differ. Use another --state-dir or inspect the existing plan.")
    if state_path.exists():
        if not path.exists():
            raise RuntimeError("Existing state.json has no frozen manifest; inspect it before restarting.")
        state = read_json(state_path)
        job_ids = {job["id"] for job in manifest["jobs"]}
        gpu_ids = {str(job["gpu"]) for job in manifest["jobs"]}
        if (state.get("schema_version") != 1 or not set(state.get("jobs", {})) <= job_ids
                or not set(state.get("gpus", {})) <= gpu_ids):
            raise RuntimeError("Existing queue state does not match the selected jobs/GPU.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--loss", choices=("detach", "rms", "both"),
                        help="Default: llama runs detach then rms; qwen runs rms.")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--state-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Print the plan; no GPU checks, files, or training.")
    modes.add_argument("--check-only", action="store_true", help="Check CUDA/BF16/packages/model config; no training.")
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be a nonnegative physical GPU index")
    if args.state_dir is None:
        loss_tag = "_".join(selected_losses(args))
        args.state_dir = ROOT / "logs" / f"massive_gap_{args.model}_{loss_tag}_gpu{args.gpu}"
    return args


def main(argv=None):
    args = parse_args(argv)
    manifest = build_manifest(args)
    for index, job in enumerate(manifest["jobs"], 1):
        print(f"{index}. {job['model']} / MASSIVE / alternative / {job['loss']} "
              "/ layer -1 / BF16 / seed 42 / 100000 steps", flush=True)
        print("   -> final IN retrieval + MASSIVE Slot F1/EM (IN+OUT), then the next run", flush=True)
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
        MassiveGapQueue(manifest, state_dir).run()


if __name__ == "__main__":
    main()
