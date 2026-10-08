"""Run the three seed-42 final-layer experiments, evaluating each before the next.

Uses the existing final-adapter and evaluation artifact checks. No local queue
manifest, server-specific Python path, or running process is imported.
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
from config import parse_args as parse_training_args
from scripts.mt_retrieval_multiseed_queue import ExperimentQueue, atomic_json
from scripts.wmt23_pipeline import config_matches, file_sha256, find_completed, read_json

MANIFEST_SHA256 = "c36ffdebbe1d3e16e23115e33003e7d14867e0ee6a8d81cfee61dafe53fab0eb"
EXPERIMENTS = (
    ("massive_qwen_infonce_then_sft", "Qwen/Qwen3.5-2B", "massive", "infonce"),
    ("mt_llama_gap_then_sft", "meta-llama/Llama-3.2-1B-Instruct", "wmt23", "gap_distance_infonce"),
    ("mt_qwen_gap_then_sft", "Qwen/Qwen3.5-2B", "wmt23", "gap_distance_infonce"),
)


def build_manifest(args):
    data_dir = args.data_dir.expanduser().resolve()
    data_path = data_dir / "manifest.json"
    if file_sha256(data_path) != args.manifest_sha256:
        raise ValueError("MT manifest hash mismatch. Copy the prepared data, or explicitly supply --manifest-sha256.")
    data = read_json(data_path)
    if data["corpus_profile"] != "alma_ja_opus" or data["data_seed"] != 42:
        raise ValueError("These experiments require the alma_ja_opus profile with data seed 42.")
    result_roots = {"massive": ROOT / "results", "wmt23": ROOT / "results_wmt23"}
    jobs = []
    for job_id, model, task, loss in EXPERIMENTS:
        script = f"scripts/{task}_transfer_only.sh"
        # Clear interactive overrides, then set every supported scientific option.
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("ALIGNMENT_", "WMT23_", "TRAIN_", "TRAINING_", "WANDB_"))
               and key not in ("MODEL_NAME", "OUTPUT_ROOT", "DOWNSTREAM_MICRO_BATCH_SIZE")}
        env.update(DRY_RUN="1", PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES=str(args.gpu),
                   MODEL_NAME=model, ALIGNMENT_LOSS=loss, ALIGNMENT_BATCHING="same_pair",
                   ALIGNMENT_HIDDEN_STATE_LAYER="-1", TRAINING_SEED="42", TRAIN_EVAL_LANGUAGE_SCOPE="in",
                   WMT23_DATA_DIR=str(data_dir), WMT23_MANIFEST_SHA256=args.manifest_sha256,
                   WMT23_CORPUS_PROFILE="alma_ja_opus", WMT23_DOWNSTREAM_SAMPLING="balanced_mixed",
                   WANDB_MODE=args.wandb_mode, OUTPUT_ROOT=str(result_roots[task]))
        preview = subprocess.run(["bash", script, "contrastive_then_transfer"], cwd=ROOT,
                                 env=env, capture_output=True, text=True, check=True)
        lines = [line for line in preview.stdout.splitlines() if line.startswith("CUDA_VISIBLE_DEVICES=")]
        if len(lines) != 1:
            raise ValueError(f"Expected one training command from {script}.")
        command = shlex.split(lines[0])
        with patch.object(sys, "argv", ["main.py", *command[command.index("main.py") + 1:]]):
            config = vars(parse_training_args())
        expected = dict(model_name=model, downstream_task=task, alignment_loss=loss,
                        training_type="contrastive_then_transfer", alignment_hidden_state_layer=-1,
                        quantization_compute_dtype="bfloat16", training_seed=42, num_steps=100000,
                        alignment_batching="same_pair", batch_size=16, eval_language_scope="in",
                        downstream_micro_batch_size=8 if task == "wmt23" and model.startswith("Qwen/") else 0)
        if any(config[key] != value for key, value in expected.items()):
            raise ValueError(f"Launcher settings changed for {job_id}: expected {expected}")
        config.pop("wandb_run_name")  # Timestamp changes on every launcher preview.
        jobs.append(dict(id=job_id, model=model, mode="contrastive_then_transfer", loss=loss,
                         gpu=args.gpu, steps=100000, script=script, config=config,
                         results_root=str(result_roots[task]), evaluation_language_scopes=["in"]))
    pairs = ("de-en", "cs-en", "en-ja")
    return dict(
        schema_version=1, python=sys.executable, comet_python=str(args.comet_python.expanduser().absolute()),
        results_root=str(result_roots["wmt23"]), data_dir=str(data_dir), manifest_sha256=args.manifest_sha256,
        wandb_mode=args.wandb_mode, jobs=jobs, adopted={},
        evaluation_id="remote_final_mt", evaluation_language_scopes=["in"],
        max_new_tokens=16384, wmt23_batch_size=16, wmt_eos_policy="generation_config_plus_tokenizer_v1",
        test_counts={"in": {key: data["counts"]["test"][key]
                            for lang in ("de", "cs", "ja") for key in (f"en-{lang}", f"{lang}-en")}},
        retrieval=dict(evaluation_id="remote_final_retrieval", language_scopes=["in"],
                       batch_size=16, chunk_size=256, pair_counts={pair: 2000 for pair in pairs},
                       pair_file_sha256={pair: data["files"][data["opus_files"][pair]["test"]]["sha256"]
                                         for pair in pairs}),
        massive_post_training_evaluation=dict(
            enabled=True, evaluation_id="remote_final_massive", language_scopes=["in", "out"],
            alignment_language_scopes=["in"], alignment_batch_size=16, massive_batch_size=16,
            retrieval_chunk_size=256, max_new_tokens=128, eval_sample_log_limit=64,
            save_alignment_embeddings=False, save_alignment_sample_metrics=True,
            pair_counts={"in": {pair: 2000 for pair in ("en-es", "en-ja", "en-ko")}},
            language_counts={"in": {lang: 2974 for lang in ("en", "ko", "ja", "es")},
                             "out": {lang: 2974 for lang in ("fr", "de", "it")}}),
    )


def preflight(manifest, gpu):
    """Check data integrity and both environments before spending time training."""
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            package, expected = line.split("==")
            package = package.split("[")[0]
            actual = version(package).split("+")[0]
            if actual != expected:
                raise RuntimeError(f"{package}=={actual}; install requirements.txt (expected {expected}).")
    from data_utils import WMT23Dataset
    WMT23Dataset.validate_prepared_manifest(argparse.Namespace(**manifest["jobs"][1]["config"]))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    subprocess.run([sys.executable, "-c",
                    "import torch, peft, bitsandbytes; "
                    "from transformers import AutoConfig; "
                    "assert torch.cuda.is_available(), 'Training requires CUDA'; "
                    "assert torch.cuda.is_bf16_supported(), 'Training requires BF16 support'; "
                    "[AutoConfig.from_pretrained(m) for m in "
                    "['meta-llama/Llama-3.2-1B-Instruct', 'Qwen/Qwen3.5-2B']]; "
                    "print('Training environment/model config access: OK')"], env=env, check=True)
    subprocess.run([manifest["comet_python"], "-c",
                    "import comet, torch; "
                    "assert torch.cuda.is_available(), 'COMET environment requires CUDA'; "
                    "print('COMET environment: OK')"], env=env, check=True)


class RemoteQueue(ExperimentQueue):
    def environment(self, gpu):
        env = super().environment(gpu)
        env.update(WMT23_DATA_DIR=self.manifest["data_dir"], WMT23_CORPUS_PROFILE="alma_ja_opus",
                   WMT23_DOWNSTREAM_SAMPLING="balanced_mixed", WANDB_MODE=self.manifest["wandb_mode"])
        return env

    def refresh_summary(self):
        # state.json and per-job logs are the portable progress record.
        pass

    def write_retrieval_report(self):
        # Metrics stay with each run, without depending on a local report tree.
        pass

    def train_job(self, gpu, job):
        if find_completed(job["results_root"], job) is None:
            model_root = Path(job["results_root"]) / job["model"].replace("/", "__")
            for path in model_root.glob("*/experiment_config.json"):
                if config_matches(job["config"], read_json(path)):
                    raise RuntimeError(f"An incomplete matching run exists: {path.parent}. "
                                       "Inspect it before restarting; optimizer state is not saved.")
        super().train_job(gpu, job)

    def run(self):
        gpu = self.manifest["jobs"][0]["gpu"]
        self.update(status="running", pid=os.getpid())
        try:
            self.reserve_gpu(gpu)
            for job in self.manifest["jobs"]:
                self.train_job(gpu, job)
                if job["config"]["downstream_task"] == "massive":
                    self.evaluate_massive_job(gpu, job)
                else:
                    self.retrieve_job(gpu, job)
                    self.evaluate(gpu, jobs=[job])
            self.update(status="completed")
        except Exception as error:
            self.update(status="failed", error=str(error))
            raise
        finally:
            self.release_gpus()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/wmt23_alma_ja_opus")
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--comet-python", type=Path, default=ROOT / ".venv-comet22/bin/python")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "logs/remote_three_experiments")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Print the plan; no GPU checks, files, or training.")
    modes.add_argument("--check-only", action="store_true", help="Check data, environments and model config access; no training.")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be a nonnegative physical GPU index")
    manifest = build_manifest(args)
    for index, job in enumerate(manifest["jobs"], 1):
        print(f"{index}. {job['model']} / {job['config']['downstream_task']} / {job['loss']} then SFT "
              "/ layer -1 / BF16 / seed 42 / 100000 steps", flush=True)
        print("   -> In Retrieval + " + ("Slot F1/EM (In+Out)" if index == 1 else "BLEU/COMET (In)"), flush=True)
    if args.dry_run:
        return
    preflight(manifest, args.gpu)
    if args.check_only:
        print("Preflight passed. No training started.", flush=True)
        return
    state_dir = args.state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    with (state_dir / "controller.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = state_dir / "manifest.json"
        if path.is_file() and read_json(path) != manifest:
            raise RuntimeError("Saved queue settings differ. Inspect the state directory before changing the experiment.")
        atomic_json(path, manifest)
        RemoteQueue(manifest, state_dir).run()


if __name__ == "__main__":
    main()
