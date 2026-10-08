"""Run a frozen WMT23 grid with full or independently scheduled stages.

Existing training processes can be adopted without interruption. Only the old
queue controllers are replaced during installation; this worker never kills a
training process. A failed stage blocks all subsequent stages.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comet22_scoring import MODEL_ID, CHECKPOINT_SHA256, file_sha256
from config import parse_args as parse_training_args


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def count_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and isinstance(json.loads(line), dict))


def process_identity(pid):
    """Start time prevents mistaking a reused Linux PID for the adopted process."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]
    except FileNotFoundError:
        return None


def config_matches(expected, actual):
    for key, value in expected.items():
        if key in ("wandb_run_name", "output_root"):
            continue
        if key == "peft_target_modules" and value is None:
            # main.py resolves the model-specific default after parsing.
            continue
        other = actual.get(key)
        if key.endswith("_data_dir") and value is not None and other is not None:
            if Path(value).resolve() == Path(other).resolve():
                continue
        if other != value:
            return False
    return True


def completed_run(path, job):
    path = Path(path)
    required = ("experiment_config.json", "trainer_state.json", "run_metadata.json", "adapter_config.json")
    if not all((path / name).is_file() for name in required):
        return False
    config = read_json(path / "experiment_config.json")
    state = read_json(path / "trainer_state.json")
    metadata = read_json(path / "run_metadata.json")
    return (config_matches(job["config"], config)
            and state.get("global_step") == job["steps"]
            and config.get("checkpoint_global_step") == job["steps"]
            and metadata.get("status") == "completed"
            and any((path / name).is_file() and (path / name).stat().st_size > 0
                    for name in ("adapter_model.safetensors", "adapter_model.bin")))


def find_completed(results_root, job):
    model_dir = Path(results_root) / job["model"].replace("/", "__")
    matches = [path for path in sorted(model_dir.glob("*")) if path.is_dir() and completed_run(path, job)]
    return matches[-1] if matches else None


def evaluation_counts(manifest, job):
    """Resolve a job's frozen scopes, including jobs adopted from a both-scope run."""
    scopes = job.get("evaluation_language_scopes", manifest.get(
        "evaluation_language_scopes", list(manifest["test_counts"])))
    if not isinstance(scopes, list) or not scopes or len(set(scopes)) != len(scopes):
        raise ValueError(f"Invalid evaluation scopes: {scopes!r}")
    if any(scope not in ("in", "out") or scope not in manifest["test_counts"] for scope in scopes):
        raise ValueError(f"Unknown evaluation scopes: {scopes!r}")
    return {scope: manifest["test_counts"][scope] for scope in ("in", "out") if scope in scopes}


def generation_complete(output_dir, run, expected_counts, max_new_tokens,
                        batch_size=1, eos_policy=None):
    output_dir = Path(output_dir)
    metadata_path = output_dir / "test" / "evaluation_metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = read_json(metadata_path)
    scopes = [scope for scope in ("in", "out") if scope in expected_counts]
    if not scopes or len(scopes) != len(expected_counts):
        return False
    if not (metadata.get("status") == "completed" and metadata.get("tasks") == ["wmt23"]
            and Path(metadata.get("checkpoint_path", "")).resolve() == Path(run).resolve()
            and metadata.get("language_scopes") == scopes
            and metadata.get("wmt23_metric") == "sacrebleu"
            and metadata.get("wmt23_batch_size") == batch_size
            and (eos_policy is None or metadata.get("wmt_eos_policy") == eos_policy)
            and metadata.get("wmt23_max_new_tokens") == max_new_tokens):
        return False
    for scope, counts in expected_counts.items():
        folder = output_dir / "test" / scope
        path = folder / "wmt23_metrics.json"
        if not path.is_file():
            return False
        metrics = read_json(path)
        if metrics.get("status") != "scored" or metrics.get("metric") != "sacrebleu":
            return False
        if {key: row["num_examples"] for key, row in metrics["by_language"].items()} != counts:
            return False
        for direction, count in counts.items():
            predictions = folder / f"wmt23_predictions.{direction}.jsonl"
            if not predictions.is_file() or count_jsonl(predictions) != count:
                return False
    return True


def comet_complete(folder, counts):
    folder = Path(folder)
    path = folder / "wmt23_comet22_metrics.json"
    if not path.is_file() or not (folder / "wmt23_comet22_scores.jsonl").is_file():
        return False
    report = read_json(path)
    metadata = report.get("scorer_metadata", {})
    if not (report.get("status") == "scored" and metadata.get("model_id") == MODEL_ID
            and metadata.get("checkpoint_sha256") == CHECKPOINT_SHA256):
        return False
    if {key: row["num_examples"] for key, row in report["by_language"].items()} != counts:
        return False
    if report.get("num_examples") != sum(counts.values()) or count_jsonl(folder / "wmt23_comet22_scores.jsonl") != sum(counts.values()):
        return False
    files = report.get("prediction_files", [])
    expected = {str((folder / f"wmt23_predictions.{key}.jsonl").absolute()) for key in counts}
    return (len(files) == len(expected) and {row["path"] for row in files} == expected
            and all(Path(row["path"]).is_file() and file_sha256(row["path"]) == row["sha256"] for row in files))


DEFAULT_STAGES = ("priority_training", "mt_evaluation", "later_training")
ALLOWED_STAGES = (DEFAULT_STAGES, ("priority_training", "mt_evaluation"),
                  ("priority_training", "later_training"), ("later_training",), ("mt_evaluation",))


def evaluation_dependency_complete(dependency):
    """Release training only after all required scopes have verified MT scores."""
    directory = Path(dependency["state_dir"])
    manifest_path = directory / "manifest.json"
    if file_sha256(manifest_path) != dependency["manifest_sha256"]:
        raise RuntimeError("Prerequisite MT manifest changed; later training remains blocked.")
    manifest = read_json(manifest_path)
    state = read_json(directory / "state.json")
    status = state.get("status")
    if status in ("prepared", "running"):
        return False
    if status != "completed":
        raise RuntimeError(f"Prerequisite MT queue is {status!r}; later training cannot start.")
    jobs = [job for job in manifest["jobs"] if job["priority"]]
    if not jobs:
        raise RuntimeError("Prerequisite MT queue has no priority evaluations.")
    for job in jobs:
        result = state.get("jobs", {}).get(job["id"], {})
        run = result.get("run")
        if result.get("evaluation_status") != "completed" or not run or not completed_run(run, job):
            raise RuntimeError(f"Unverified prerequisite checkpoint/evaluation: {job['id']}")
        output = Path(run) / "evaluations" / manifest["evaluation_id"]
        counts = evaluation_counts(manifest, job)
        if not generation_complete(output, run, counts, manifest["max_new_tokens"],
                                   manifest.get("wmt23_batch_size", 1), manifest.get("wmt_eos_policy")):
            raise RuntimeError(f"Incomplete prerequisite MT generation/BLEU: {job['id']}")
        for scope, scope_counts in counts.items():
            if not comet_complete(output / "test" / scope, scope_counts):
                raise RuntimeError(f"Incomplete prerequisite COMET-22: {job['id']} / {scope}")
    return True


def run_phases(gpus, train_priority, evaluate, train_later, stages=DEFAULT_STAGES):
    """A global barrier separates stages; exceptions prevent later stages."""
    if tuple(stages) not in ALLOWED_STAGES:
        raise ValueError(f"Unsupported pipeline stages: {stages}")
    actions = dict(zip(DEFAULT_STAGES, (train_priority, evaluate, train_later)))
    for stage in stages:
        action = actions[stage]
        with ThreadPoolExecutor(max_workers=len(gpus)) as workers:
            futures = [workers.submit(action, gpu) for gpu in gpus]
            for future in as_completed(futures):
                future.result()


class Pipeline:
    def __init__(self, manifest, state_dir):
        self.manifest = manifest
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.mutex = threading.RLock()
        self.summary_lock = threading.Lock()
        self.gpu_locks = []
        self.reserved_gpus = set()
        self.failed = threading.Event()
        state_path = self.state_dir / "state.json"
        self.state = read_json(state_path) if state_path.is_file() else {"schema_version": 1, "jobs": {}, "gpus": {}}

    def update(self, *, gpu=None, job=None, **fields):
        with self.mutex:
            target = self.state
            if job is not None:
                target = self.state["jobs"].setdefault(job["id"], {})
            elif gpu is not None:
                target = self.state["gpus"].setdefault(str(gpu), {})
            target.update(fields)
            self.state["updated_at"] = datetime.now().astimezone().isoformat()
            temporary = self.state_dir / "state.json.tmp"
            temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.state_dir / "state.json")
            print(datetime.now().astimezone().isoformat(), f"gpu={gpu} job={job['id'] if job else '-'}", fields, flush=True)

    @contextmanager
    def lock(self, path):
        with Path(path).open("a") as handle:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if self.failed.is_set():
                        raise RuntimeError("Another worker failed while waiting for the GPU lock.")
                    time.sleep(15)
            yield handle

    def environment(self, gpu):
        env = os.environ.copy()
        # Avoid inheriting interactive launcher overrides into a frozen grid.
        for key in list(env):
            if key.startswith(("EVAL_", "COMET22_", "WMT23_")) or key in (
                "DRY_RUN", "MODEL_NAME", "ALIGNMENT_LOSS", "DOWNSTREAM_MICRO_BATCH_SIZE",
                "TRAIN_SAMPLE_LOG_INTERVAL", "TRAIN_SAMPLE_LOG_LIMIT", "ALIGNMENT_BATCHING",
                "TRAINING_SEED", "TRAIN_EVAL_LANGUAGE_SCOPE", "ALIGNMENT_HIDDEN_STATE_LAYER",
                "OUTPUT_ROOT", "WANDB_PROJECT", "WANDB_MODE", "PYTHON_BIN",
            ):
                del env[key]
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHON_BIN=self.manifest["python"],
                   PYTORCH_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        return env

    def execute(self, command, env, log, lock_handle):
        with Path(log).open("a") as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                       pass_fds=(lock_handle.fileno(),))
            gpu = int(env["CUDA_VISIBLE_DEVICES"])
            self.update(gpu=gpu, child_pid=process.pid, child_start_time=process_identity(process.pid), command=command, log=str(log))
            returncode = process.wait()
            self.update(gpu=gpu, child_pid=None, child_start_time=None)
            if returncode != 0:
                raise RuntimeError(f"Command failed; see {log}")

    def train_job(self, gpu, job):
        if self.failed.is_set():
            raise RuntimeError("Another worker failed; no new training will start.")
        results_root = job.get("results_root", self.manifest["results_root"])
        pinned = self.state["jobs"].get(job["id"], {}).get("run")
        existing = (Path(pinned) if pinned and completed_run(pinned, job)
                    else find_completed(results_root, job))
        if existing:
            self.update(job=job, training_status="completed", run=str(existing), reused=True)
            return
        env = self.environment(gpu)
        env.update(MODEL_NAME=job["model"], ALIGNMENT_LOSS=job["loss"],
                   OUTPUT_ROOT=job["config"]["output_root"],
                   TRAIN_EVAL_LANGUAGE_SCOPE=job["config"].get("eval_language_scope", "in"),
                   ALIGNMENT_HIDDEN_STATE_LAYER=str(job["config"].get("alignment_hidden_state_layer", -1)),
                   WMT23_MANIFEST_SHA256=self.manifest["manifest_sha256"], TRAINING_SEED=str(job["config"]["training_seed"]))
        preview = subprocess.run(["bash", job["script"], job["mode"]], cwd=ROOT,
                                 env={**env, "DRY_RUN": "1"}, capture_output=True, text=True, check=True)
        lines = [line for line in preview.stdout.splitlines() if line.startswith("CUDA_VISIBLE_DEVICES=")]
        if len(lines) != 1:
            raise ValueError("Expected exactly one training command from the selected model/loss/mode.")
        command = shlex.split(lines[0])[1:]
        # Parse the actual command to catch changed training scripts before starting a model.
        with self.mutex, patch.object(sys, "argv", ["main.py", *command[command.index("main.py") + 1:]]):
            actual = vars(parse_training_args())
        if not config_matches(job["config"], actual):
            raise ValueError(f"Training launcher changed the frozen settings for {job['id']}.")
        log = ROOT / "logs" / f"{actual['wandb_run_name']}.log"
        self.update(job=job, training_status="running", log=str(log))
        self.refresh_summary()
        with self.lock(ROOT / "logs" / f"train-gpu-{gpu}.lock") as handle:
            self.execute(command, env, log, handle)
        run = find_completed(results_root, job)
        if run is None:
            raise RuntimeError(f"Training exited without a verified final adapter: {job['id']}")
        self.update(job=job, training_status="completed", run=str(run), reused=False)
        self.refresh_summary()

    def reserve_gpu(self, gpu):
        if gpu in self.reserved_gpus:
            return
        adopted = self.manifest["adopted"].get(str(gpu))
        # On restart, wait for a child from this pipeline instead of duplicating it.
        previous = dict(self.state["gpus"].get(str(gpu), {}))
        existing_child = (previous.get("child_pid") and
                          process_identity(previous["child_pid"]) == previous.get("child_start_time"))
        adopted_pids = [item["pid"] for item in adopted["processes"]
                        if process_identity(item["pid"]) == item["start_time"]] if adopted else []
        self.update(gpu=gpu, status="waiting_for_existing_child" if existing_child else "waiting_for_adopted_training",
                    waiting_for_pids=[previous["child_pid"]] if existing_child else adopted_pids)
        while previous.get("child_pid") and process_identity(previous["child_pid"]) == previous.get("child_start_time"):
            if self.failed.is_set():
                raise RuntimeError("Another worker failed; the existing child is left running.")
            time.sleep(15)
        if adopted:
            while any(process_identity(item["pid"]) == item["start_time"] for item in adopted["processes"]):
                if self.failed.is_set():
                    raise RuntimeError("Another worker failed; adopted training is left running.")
                time.sleep(15)
            job = next(item for item in self.manifest["jobs"] if item["id"] == adopted["job_id"])
            if not completed_run(adopted["run"], job):
                raise RuntimeError(f"Adopted training ended without a complete final adapter: {adopted['run']}")
        # Old training children inherited the old controller's grid lock.
        # Acquire it only after those children exit, and retain it across phases.
        handle = (ROOT / "logs" / f"wmt23-full-grid-gpu-{gpu}.lock").open("a")
        fcntl.flock(handle, fcntl.LOCK_EX)
        with self.mutex:
            self.gpu_locks.append(handle)
            self.reserved_gpus.add(gpu)

    def priority(self, gpu):
        try:
            self.update(gpu=gpu, phase="priority_training")
            self.reserve_gpu(gpu)
            self.update(gpu=gpu, status="running")
            for job in self.manifest["jobs"]:
                if job["gpu"] == gpu and job["priority"]:
                    self.train_job(gpu, job)
            self.update(gpu=gpu, status="priority_complete_waiting_for_all_gpus")
        except Exception as error:
            self.failed.set()
            self.update(gpu=gpu, status="failed", error=str(error))
            self.update(status="failed", error=str(error))
            raise

    def evaluate(self, gpu, *, jobs=None):
        active_job = None
        try:
            self.update(gpu=gpu, phase="mt_evaluation")
            self.reserve_gpu(gpu)
            self.update(gpu=gpu, phase="mt_evaluation", status="running")
            for job in self.evaluation_jobs(gpu) if jobs is None else jobs:
                active_job = job
                if self.failed.is_set():
                    raise RuntimeError("Another evaluation in this queue failed.")
                pinned = self.state["jobs"].get(job["id"], {}).get("run")
                run = Path(pinned) if pinned else find_completed(job.get("results_root", self.manifest["results_root"]), job)
                if run is None or not completed_run(run, job):
                    raise RuntimeError(f"No verified final adapter for evaluation: {job['id']}")
                self.update(job=job, run=str(run), training_status="completed")
                output = run / "evaluations" / job.get("evaluation_id", self.manifest["evaluation_id"])
                counts = evaluation_counts(self.manifest, job)
                language_scope = "both" if len(counts) == 2 else next(iter(counts))
                batch_size = self.manifest.get("wmt23_batch_size", 1)
                eos_policy = self.manifest.get("wmt_eos_policy")
                env = self.environment(gpu)
                env.update(EVAL_TASKS="wmt23", EVAL_SPLIT="test", EVAL_LANGUAGE_SCOPE=language_scope,
                           EVAL_OUTPUT_DIR=str(output), EVAL_WMT23_METRIC="sacrebleu", EVAL_COMET22="false",
                           EVAL_WMT23_MAX_NEW_TOKENS=str(self.manifest["max_new_tokens"]),
                           EVAL_WMT23_BATCH_SIZE=str(batch_size))
                log = self.state_dir / f"{job['id']}.evaluation.log"
                self.update(job=job, evaluation_status="running", evaluation_dir=str(output), evaluation_log=str(log),
                            evaluation_language_scopes=list(counts))
                with self.lock(ROOT / "logs" / f"train-gpu-{gpu}.lock") as handle:
                    if not generation_complete(output, run, counts, self.manifest["max_new_tokens"], batch_size, eos_policy):
                        self.execute(["bash", "scripts/wmt23_eval.sh", str(run)], env, log, handle)
                    if not generation_complete(output, run, counts, self.manifest["max_new_tokens"], batch_size, eos_policy):
                        raise RuntimeError(f"Incomplete MT generation/BLEU artifacts: {run}")
                    for scope in counts:
                        folder = output / "test" / scope
                        if not comet_complete(folder, counts[scope]):
                            self.execute([self.manifest["comet_python"], "scripts/score_comet22.py",
                                          "--prediction_dir", str(folder), "--gpus", "1", "--batch_size", "16"], env, log, handle)
                        if not comet_complete(folder, counts[scope]):
                            raise RuntimeError(f"Incomplete COMET-22 artifacts: {folder}")
                self.update(job=job, evaluation_status="completed")
                active_job = None
                self.refresh_summary()
            self.update(gpu=gpu, status="mt_evaluation_complete_waiting_for_all_gpus")
        except Exception as error:
            self.failed.set()
            if active_job is not None:
                self.update(job=active_job, evaluation_status="failed", evaluation_error=str(error))
            self.update(gpu=gpu, status="failed", error=str(error))
            self.update(status="failed", error=str(error))
            self.refresh_summary()
            raise

    def claim_evaluation(self, gpu):
        """Claim one ready final adapter exactly once across GPU workers."""
        with self.mutex:
            remaining = False
            for job in self.manifest["jobs"]:
                if not job["priority"]:
                    continue
                state = self.state["jobs"].get(job["id"], {})
                status = state.get("evaluation_status", "queued")
                if status == "completed":
                    continue
                remaining = True
                if status in ("claimed", "running"):
                    continue
                if status == "failed":
                    raise RuntimeError(f"Evaluation previously failed: {job['id']}")
                pinned = state.get("run")
                run = Path(pinned) if pinned else find_completed(self.manifest["results_root"], job)
                if run is None or not completed_run(run, job):
                    continue
                self.update(job=job, evaluation_status="claimed", evaluation_gpu=gpu,
                            run=str(run), training_status="completed")
                return job, True
            return None, remaining

    def evaluation_jobs(self, gpu):
        if not self.manifest.get("evaluation_work_stealing", False):
            yield from (job for job in self.manifest["jobs"] if job["gpu"] == gpu and job["priority"])
            return
        # A restarted worker waits for its old child in reserve_gpu, then can
        # validate/reuse its output instead of leaving a permanently claimed job.
        with self.mutex:
            for job in self.manifest["jobs"]:
                state = self.state["jobs"].get(job["id"], {})
                if state.get("evaluation_gpu") == gpu and state.get("evaluation_status") in ("claimed", "running"):
                    self.update(job=job, evaluation_status="queued")
        while not self.failed.is_set():
            job, remaining = self.claim_evaluation(gpu)
            if job is not None:
                yield job
            elif not remaining:
                return
            else:
                time.sleep(15)
        raise RuntimeError("Another evaluation in this queue failed.")

    def refresh_summary(self):
        command = self.manifest.get("summary_command")
        if not command:
            return
        with self.summary_lock:
            try:
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            except OSError as error:
                self.update(summary_status="failed", summary_error=str(error))
                return
            if result.returncode:
                self.update(summary_status="failed", summary_error=result.stderr[-4000:])
            else:
                self.update(summary_status="updated", summary_error=None)

    def later(self, gpu):
        try:
            self.reserve_gpu(gpu)
            self.update(gpu=gpu, phase="later_training", status="running")
            for job in self.manifest["jobs"]:
                if job["gpu"] == gpu and not job["priority"]:
                    self.train_job(gpu, job)
            self.update(gpu=gpu, status="completed")
        except Exception as error:
            self.failed.set()
            self.update(gpu=gpu, status="failed", error=str(error))
            self.update(status="failed", error=str(error))
            raise

    def run(self):
        stages = self.manifest.get("stages", DEFAULT_STAGES)
        self.update(status="running", pid=os.getpid(), phase_order=list(stages))
        try:
            dependency = self.manifest.get("wait_for_evaluation")
            if dependency:
                # Wait before taking GPU locks so the active evaluator can finish.
                self.update(status="waiting_for_mt_evaluation", evaluation_prerequisite=dependency)
                while not evaluation_dependency_complete(dependency):
                    time.sleep(30)
                self.update(status="running", evaluation_prerequisite_verified_at=datetime.now().astimezone().isoformat())
            gpus = self.manifest.get("evaluation_gpus", sorted({job["gpu"] for job in self.manifest["jobs"]}))
            run_phases(gpus, self.priority, self.evaluate, self.later, stages)
            self.update(status="completed")
        except Exception as error:
            self.update(status="failed", error=str(error))
            raise
        finally:
            for handle in self.gpu_locks:
                handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state_dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    if manifest.get("superseded_by"):
        raise ValueError(f"This queue was replaced by {manifest['superseded_by']}; refusing to launch retired jobs.")
    if args.dry_run:
        for phase in manifest.get("stages", DEFAULT_STAGES):
            for job in manifest["jobs"]:
                if job["priority"] == (phase != "later_training"):
                    done = find_completed(manifest["results_root"], job)
                    print(phase, f"gpu={job['gpu']}", job["id"], "completed" if done else "pending/running", done or "")
        return
    if file_sha256(manifest["data_manifest"]) != manifest["manifest_sha256"]:
        raise ValueError("Prepared WMT23 data manifest changed.")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "controller.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Pipeline(manifest, args.state_dir).run()


if __name__ == "__main__":
    main()
