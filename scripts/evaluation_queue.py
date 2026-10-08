"""Evaluate completed runs after the selected training queues exit.

Two independent workers each expose one GPU and process disjoint run lists.
Waiting and planning use only the standard library and never load a model.
"""

import argparse
from contextlib import contextmanager
from datetime import date, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT = PROJECT_ROOT / "logs" / "evaluation_queue"
STEP_BUDGETS = {
    "transfer_only": 50000, "contrastive_only": 50000,
    "alternative": 100000, "contrastive_then_transfer": 100000,
}
TRAINING_SCRIPTS = {
    # Keep old names too: queues launched before a rename may still be alive.
    "main.py", "scripts/transfer_only.sh", "scripts/contrastive_only.sh",
    "scripts/massive_transfer_only.sh", "scripts/massive_contrastive_only.sh",
    "scripts/wmt25_transfer_only.sh", "scripts/wmt25_contrastive_only.sh",
    "scripts/alternative.sh", "scripts/contrastive_then_transfer.sh",
    "scripts/layer_probe.sh", "scripts/lr_sweep.sh",
}
EVAL_DEFAULTS = {
    "EVAL_SPLIT": "test", "EVAL_LANGUAGE_SCOPE": "both",
    "EVAL_TASKS": "alignment massive", "EVAL_ALIGNMENT_BATCH_SIZE": "16",
    "EVAL_MASSIVE_BATCH_SIZE": "16", "EVAL_RETRIEVAL_CHUNK_SIZE": "256",
    "EVAL_MAX_NEW_TOKENS": "128", "EVAL_SAMPLE_LOG_LIMIT": "64",
    "EVAL_SAVE_ALIGNMENT_EMBEDDINGS": "false",
    "EVAL_SAVE_ALIGNMENT_SAMPLE_METRICS": "true",
}


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def lock_file(path, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def inspect_run(run, results_root):
    """Require successful final saving and the method's exact step budget."""
    try:
        metadata = read_json(run / "run_metadata.json")
        if metadata.get("status") != "completed":
            return None, f"training status: {metadata.get('status', 'missing')}"
        config = read_json(run / "experiment_config.json")
        state = read_json(run / "trainer_state.json")
        read_json(run / "train_results.json")
        read_json(run / "adapter_config.json")
        training_type = config.get("training_type")
        expected = STEP_BUDGETS.get(training_type)
        if expected is None:
            return None, f"unsupported training_type: {training_type}"
        if config.get("num_steps") != expected or state.get("global_step") != expected:
            return None, (f"requires {expected} steps; configured={config.get('num_steps')}, "
                          f"completed={state.get('global_step')}")
        saved_step = config.get("checkpoint_global_step")
        if saved_step is not None and saved_step != expected:
            return None, f"root adapter step is {saved_step}, expected {expected}"
        weights = next((run / name for name in (
            "adapter_model.safetensors", "adapter_model.bin",
        ) if (run / name).is_file() and (run / name).stat().st_size > 0), None)
        if weights is None:
            return None, "missing or empty final adapter weights"
        return {
            "run": str(run.relative_to(results_root)), "checkpoint_path": str(run.resolve()),
            "training_type": training_type, "global_step": expected,
            "model_name": config.get("model_name"),
            "alignment_loss": config.get("alignment_loss", "infonce"),
            "alignment_batching": config.get("alignment_batching", "mixed"),
            "quantization_compute_dtype": config.get("quantization_compute_dtype"),
            "completed_at": metadata.get("completed_at"),
        }, None
    except (OSError, ValueError) as error:
        return None, f"invalid completion evidence: {error}"


def matching_evaluation(run, entry, settings):
    """Find a successful evaluation of this final model with matching settings."""
    scopes = ["in", "out"] if settings["EVAL_LANGUAGE_SCOPE"] == "both" else [settings["EVAL_LANGUAGE_SCOPE"]]
    tasks = settings["EVAL_TASKS"].split()
    options = {
        "alignment_batch_size": int(settings["EVAL_ALIGNMENT_BATCH_SIZE"]),
        "massive_batch_size": int(settings["EVAL_MASSIVE_BATCH_SIZE"]),
        "retrieval_chunk_size": int(settings["EVAL_RETRIEVAL_CHUNK_SIZE"]),
        "max_new_tokens": int(settings["EVAL_MAX_NEW_TOKENS"]),
        "eval_sample_log_limit": int(settings["EVAL_SAMPLE_LOG_LIMIT"]),
        "save_alignment_embeddings": settings["EVAL_SAVE_ALIGNMENT_EMBEDDINGS"] == "true",
        "save_alignment_sample_metrics": settings["EVAL_SAVE_ALIGNMENT_SAMPLE_METRICS"] == "true",
    }
    for path in sorted((run / "evaluations").glob("**/evaluation_metadata.json")):
        try:
            metadata = read_json(path)
            if metadata.get("status") != "completed" or metadata.get("split") != settings["EVAL_SPLIT"]:
                continue
            if Path(metadata.get("checkpoint_path", "")).resolve() != run.resolve():
                continue
            if metadata.get("experiment_config", {}).get("checkpoint_global_step") != entry["global_step"]:
                continue
            if not set(scopes) <= set(metadata.get("language_scopes", [])) or not set(tasks) <= set(metadata.get("tasks", [])):
                continue
            if any(metadata.get(key) != value for key, value in options.items()):
                continue
            # Partial output cannot suppress a requested evaluation.
            for scope in scopes:
                for task in tasks:
                    metrics = read_json(path.parent / scope / f"{task}_metrics.json")
                    if not metrics or metrics != metadata.get("results", {}).get(scope, {}).get(task):
                        raise ValueError("Missing or inconsistent evaluation metrics")
            return str(path)
        except (OSError, ValueError, TypeError):
            continue
    return None


def assign_gpus(runs, priority_gpu=None, existing=()):
    """Keep existing assignments; pin priorities and balance the remaining runs."""
    assigned = {entry["run"]: entry["gpu"] for entry in existing}
    counts = [sum(gpu == device for gpu in assigned.values()) for device in (0, 1)]
    for entry in runs:
        if entry["run"] in assigned:
            entry["gpu"] = assigned[entry["run"]]
            continue
        gpu = priority_gpu if entry.get("priority") and priority_gpu is not None else min((0, 1), key=lambda g: counts[g])
        entry["gpu"] = gpu
        assigned[entry["run"]] = gpu
        counts[gpu] += 1


def scan_runs(results_root, *, completed_since=None, priority_runs=(), priority_gpu=None, skip_evaluated=False, settings=None):
    runs, excluded = [], []
    cutoff = date.fromisoformat(completed_since) if completed_since else None
    settings = settings or evaluation_settings()
    priorities = {run: index for index, run in enumerate(priority_runs)}
    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        for run in sorted(model_dir.iterdir()):
            if not run.is_dir() or run.name.startswith("."):
                continue
            entry, reason = inspect_run(run, results_root)
            if entry is not None and cutoff is not None:
                try:
                    completed = datetime.fromisoformat(entry["completed_at"])
                    if completed.tzinfo is not None:
                        completed = completed.astimezone(ZoneInfo("Asia/Seoul"))
                    if completed.date() < cutoff:
                        reason = f"completed before {completed_since} (KST)"
                except (TypeError, ValueError):
                    reason = "missing or invalid completion date"
            if entry is not None and reason is None and skip_evaluated:
                previous = matching_evaluation(run, entry, settings)
                if previous:
                    reason = f"already evaluated with matching settings: {previous}"
            if reason is not None:
                excluded.append({"run": str(run.relative_to(results_root)), "reason": reason})
            else:
                entry["priority"] = entry["run"] in priorities
                runs.append(entry)
    runs.sort(key=lambda entry: (priorities.get(entry["run"], len(priorities)), entry["run"]))
    assign_gpus(runs, priority_gpu)
    return {"scanned_at": now(), "runs": runs, "excluded": excluded}


def selection_settings(args):
    return {"completed_since": getattr(args, "completed_since", None),
            "priority_runs": list(getattr(args, "priority_runs", ())),
            "priority_gpu": getattr(args, "priority_gpu", None),
            "skip_evaluated": getattr(args, "skip_evaluated", False)}


def requested_settings(args):
    return getattr(args, "settings", None) or evaluation_settings()


def selected_runs(args):
    return scan_runs(args.results_root, **selection_settings(args), settings=requested_settings(args))


def process_info(pid, proc_root=Path("/proc")):
    try:
        fields = (proc_root / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": fields[19]}
    except (OSError, ValueError, IndexError):
        return None


def process_alive(identity):
    return identity is not None and process_info(identity["pid"]) == identity


def process_gpu_ids(process, project_root):
    """Read only device identifiers, never expose a process's full environment."""
    try:
        for value in (process / "environ").read_bytes().split(b"\0"):
            if value.startswith(b"CUDA_VISIBLE_DEVICES="):
                devices = value.split(b"=", 1)[1].decode().split(",")
                if all(device.strip().isdigit() for device in devices):
                    return [int(device) for device in devices]
        # A shell's /proc environment may predate its export statement. The
        # training lock FD is inherited by its children and identifies its GPU.
        paths = {fd.resolve() for fd in (process / "fd").iterdir()}
        devices = [gpu for gpu in (0, 1)
                   if project_root / "logs" / f"train-gpu-{gpu}.lock" in paths]
        return devices or None
    except (OSError, ValueError):
        return None


def training_processes(project_root=PROJECT_ROOT, proc_root=Path("/proc")):
    # A gap between a shell queue's Python children must not count as idle.
    targets = {str((project_root / name).resolve()) for name in TRAINING_SCRIPTS}
    basenames = {Path(name).name for name in TRAINING_SCRIPTS}
    found = []
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            identity = process_info(process.name, proc_root)
            if identity is None:
                continue
            argv = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
            cwd = (process / "cwd").resolve(strict=True)
            for arg in argv[1:]:
                if Path(arg).name in basenames and str((cwd / arg).resolve()) in targets:
                    found.append({**identity, "script": Path(arg).name,
                                  "gpu_ids": process_gpu_ids(process, project_root)})
                    break
        except (OSError, ValueError):
            continue
    return sorted(found, key=lambda item: item["pid"])


def training_lock_busy(gpu):
    path = PROJECT_ROOT / "logs" / f"train-gpu-{gpu}.lock"
    if not path.exists():
        return False
    try:
        with lock_file(path, blocking=False):
            return False
    except BlockingIOError:
        return True


def training_pending():
    return bool(training_processes()) or any(training_lock_busy(gpu) for gpu in (0, 1))


def gpu_processes(gpu):
    def query(fields):
        result = subprocess.run(
            ["nvidia-smi", fields, "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return [line.split(",") for line in result.stdout.splitlines() if line.strip()]
    gpus = {int(index.strip()): uuid.strip() for index, uuid in query("--query-gpu=index,uuid")}
    if gpu not in gpus:
        raise RuntimeError(f"GPU {gpu} is not available")
    return [int(pid.strip()) for uuid, pid in query("--query-compute-apps=gpu_uuid,pid")
            if uuid.strip() == gpus[gpu]]


def notify(args, title, body):
    if args.no_notify:
        return
    request = Request(args.ntfy_url, data=body.encode("utf-8"), method="POST", headers={
        "Title": title, "Content-Type": "text/plain; charset=utf-8",
    })
    try:
        with urlopen(request, timeout=15) as response:
            receipt = json.load(response)
        print(f"[{now()}] ntfy delivered: {title} ({receipt.get('id', 'ok')})", flush=True)
        with (args.queue_dir / "notifications.jsonl").open("a") as handle:
            handle.write(json.dumps({"time": now(), "title": title, "receipt": receipt}) + "\n")
    except Exception as error:
        print(f"[{now()}] ntfy failed: {error}", flush=True)


def update_state(args, state, **changes):
    state.update(changes, updated_at=now())
    write_json(args.queue_dir / f"gpu{args.gpu}_state.json", state)


def wait_until_idle(args, state):
    idle_since, previous = None, None
    while True:
        training = training_processes()
        if getattr(args, "wait_scope", "all") == "gpu":
            training = [p for p in training
                        if p.get("gpu_ids") is None or args.gpu in p["gpu_ids"]]
        training_locked = training_lock_busy(args.gpu)
        try:
            busy, error = gpu_processes(args.gpu), None
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exception:
            busy, error = [], str(exception)
        if training or training_locked or busy or error:
            idle_since = None
            phase = "waiting_for_training" if training or training_locked else "waiting_for_gpu"
        else:
            idle_since = time.monotonic() if idle_since is None else idle_since
            if time.monotonic() - idle_since >= args.quiet_seconds:
                return
            phase = "waiting_for_quiet_period"
        update_state(args, state, status=phase, training_processes=training,
                     training_lock_busy=training_locked, gpu_processes=busy, gpu_query_error=error)
        signature = (phase, tuple(p["pid"] for p in training), training_locked, tuple(busy), error)
        if signature != previous:
            print(f"[{now()}] GPU {args.gpu}: {phase}; training={training}; GPU PIDs={busy}; error={error}", flush=True)
            previous = signature
        time.sleep(args.poll_seconds)


def evaluation_settings():
    return {name: os.environ.get(name, default) for name, default in EVAL_DEFAULTS.items()}


def save_plan(args, report, filename):
    report = {**report, "results_root": str(args.results_root), "settings": requested_settings(args),
              "selection": selection_settings(args)}
    write_json(args.queue_dir / filename, report)
    prefix = "preview_" if filename.startswith("preview") else ""
    for gpu in (0, 1):
        paths = [entry["run"] for entry in report["runs"] if entry["gpu"] == gpu]
        (args.queue_dir / f"{prefix}runs_gpu{gpu}.txt").write_text("".join(path + "\n" for path in paths))
    return report


def load_manifest(args):
    with lock_file(args.queue_dir / ".manifest.lock"):
        path = args.queue_dir / "manifest.json"
        if path.exists():
            report = read_json(path)
            if report["results_root"] != str(args.results_root):
                raise ValueError("The queue manifest belongs to a different results root")
            if getattr(args, "refresh_manifest", False):
                latest = selected_runs(args)
                known = {entry["run"] for entry in report["runs"]}
                added = [entry for entry in latest["runs"] if entry["run"] not in known]
                assign_gpus(added, getattr(args, "priority_gpu", None), report["runs"])
                report["runs"].extend(added)
                report["excluded"] = [entry for entry in latest["excluded"]
                                      if entry["run"] not in known]
                report["scanned_at"] = latest["scanned_at"]
                return save_plan(args, report, "manifest.json")
            return report
        # Include jobs that finish while the workers are waiting.
        return save_plan(args, selected_runs(args), "manifest.json")


def run_environment(args, entry, settings):
    env = dict(os.environ)
    env.update(settings)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
        PYTHON_BIN=sys.executable, PYTHONUNBUFFERED="1",
        EVAL_OUTPUT_DIR=str(Path(entry["checkpoint_path"]) / "evaluations" / args.queue_dir.name),
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        env.pop(name, None)
    return env


def evaluate_one(args, entry, settings, state, result_path):
    log_path = args.queue_dir / "runs" / (entry["run"] + f".gpu{args.gpu}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = run_environment(args, entry, settings)
    command = ["bash", str(PROJECT_ROOT / "scripts/eval.sh"), entry["checkpoint_path"]]
    result = {**entry, "status": "running", "started_at": now(), "log": str(log_path),
              "output_dir": env["EVAL_OUTPUT_DIR"], "command": command, "settings": settings}
    result["source_sha256"] = {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        for name in ("evaluate.py", "models.py", "data_utils.py", "scripts/eval.sh")
    }
    write_json(result_path, result)
    notify(args, f"GPU {args.gpu}: evaluation started", entry["run"])
    with log_path.open("a") as log:
        log.write(f"\n[{now()}] CUDA_VISIBLE_DEVICES={args.gpu}\n{shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
        update_state(args, state, status="evaluating", current_run=entry["run"],
                     child_pid=process.pid, current_log=str(log_path),
                     training_processes=[], gpu_processes=[], gpu_query_error=None)
        try:
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    success = returncode == 0
    if success:
        try:
            metadata = read_json(Path(env["EVAL_OUTPUT_DIR"]) / settings["EVAL_SPLIT"] / "evaluation_metadata.json")
            success = metadata.get("status") == "completed"
        except (OSError, ValueError):
            success = False
    result.update(status="completed" if success else "failed", returncode=returncode, finished_at=now())
    write_json(result_path, result)
    notify(args, f"GPU {args.gpu}: evaluation {result['status']}",
           f"{entry['run']}\nExit code: {returncode}\nLog: {log_path}")
    return success


def worker(args):
    if args.gpu is None:
        raise ValueError("worker requires --gpu 0 or --gpu 1")
    with lock_file(QUEUE_ROOT / f".gpu{args.gpu}.lock", blocking=False):
        state = {"worker": process_info(os.getpid()), "gpu": args.gpu, "started_at": now(),
                 "completed": [], "failed": [], "skipped": []}
        update_state(args, state, status="starting")
        try:
            wait_until_idle(args, state)
            attempted, announced_count = set(), None
            while True:
                report = load_manifest(args)
                assigned = [entry for entry in report["runs"] if entry["gpu"] == args.gpu]
                update_state(args, state, status="ready", total=len(assigned))
                if announced_count != len(assigned):
                    notify(args, f"GPU {args.gpu}: evaluation queue ready", f"{len(assigned)} completed runs assigned.")
                    announced_count = len(assigned)
                for index, entry in enumerate(assigned, start=1):
                    if entry["run"] in attempted:
                        continue
                    attempted.add(entry["run"])
                    result_path = args.queue_dir / "outcomes" / (entry["run"] + ".json")
                    if result_path.exists() and read_json(result_path).get("status") == "completed":
                        state["completed"].append(entry["run"])
                        continue
                    wait_until_idle(args, state)
                    _, reason = inspect_run(Path(entry["checkpoint_path"]), args.results_root)
                    if reason:
                        state["skipped"].append({"run": entry["run"], "reason": reason})
                        update_state(args, state, status="skipped_incomplete_run")
                        continue
                    print(f"[{now()}] GPU {args.gpu} [{index}/{len(assigned)}] {entry['run']}", flush=True)
                    success = evaluate_one(args, entry, report["settings"], state, result_path)
                    state["completed" if success else "failed"].append(entry["run"])
                    update_state(args, state, status="between_runs", child_pid=None,
                                 current_run=None, current_log=None)
                if not getattr(args, "refresh_manifest", False):
                    break
                if not training_pending():
                    # Scan once more after the last training process exits, so
                    # a just-saved final model cannot fall between two scans.
                    final = load_manifest(args)
                    if not any(e["gpu"] == args.gpu and e["run"] not in attempted for e in final["runs"]):
                        break
                    continue
                update_state(args, state, status="waiting_for_new_results",
                             training_processes=training_processes())
                time.sleep(args.poll_seconds)
            update_state(args, state, status="finished", finished_at=now(), current_run=None)
            notify(args, f"GPU {args.gpu}: evaluation queue finished",
                   f"Completed: {len(state['completed'])}; failed: {len(state['failed'])}; skipped: {len(state['skipped'])}.")
            return int(bool(state["failed"]))
        except BaseException as error:
            update_state(args, state, status="stopped", error=str(error), stopped_at=now())
            notify(args, f"GPU {args.gpu}: evaluation queue stopped", f"{type(error).__name__}: {error}")
            raise


def print_plan(report, gpu=None):
    print(f"Completed eligible runs: {len(report['runs'])}; excluded: {len(report['excluded'])}")
    for device in ((0, 1) if gpu is None else (gpu,)):
        assigned = [entry for entry in report["runs"] if entry["gpu"] == device]
        print(f"\nGPU {device}: {len(assigned)} runs")
        for entry in assigned:
            priority = "[priority] " if entry.get("priority") else ""
            print(f"  {entry['global_step']:6d}  {priority}{entry['run']}")


def launch(args):
    with lock_file(QUEUE_ROOT / ".launch.lock", blocking=False):
        current = QUEUE_ROOT / "current.json"
        if current.exists():
            active = read_json(current)
            if any(process_alive(identity) for identity in active["workers"].values()):
                print(f"An evaluation queue is already active: {active['queue_dir']}")
                print(json.dumps(active, indent=2))
                return 0
        with lock_file(QUEUE_ROOT / ".gpu0.lock", blocking=False):
            with lock_file(QUEUE_ROOT / ".gpu1.lock", blocking=False):
                pass
        report = save_plan(args, selected_runs(args), "preview_manifest.json")
        print_plan(report)
        write_json(args.queue_dir / "queue_config.json", {
            "created_at": now(), "results_root": str(args.results_root),
            "python": sys.executable, "ntfy_url": args.ntfy_url,
            "settings": requested_settings(args), "selection": selection_settings(args),
            "scheduling": {"wait_scope": args.wait_scope,
                           "refresh_manifest": args.refresh_manifest},
            "training_processes": training_processes(),
        })
        workers = {}
        for gpu in (0, 1):
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                       "--gpu", str(gpu), "--queue-dir", str(args.queue_dir),
                       "--results-root", str(args.results_root), "--ntfy-url", args.ntfy_url,
                       "--poll-seconds", str(args.poll_seconds), "--quiet-seconds", str(args.quiet_seconds)]
            if args.no_notify:
                command.append("--no-notify")
            with (args.queue_dir / f"gpu{gpu}_worker.log").open("a") as log:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           start_new_session=True)
            workers[str(gpu)] = process_info(process.pid)
        active = {"queue_dir": str(args.queue_dir), "workers": workers, "launched_at": now()}
        write_json(current, active)
        write_json(args.queue_dir / "workers.json", active)
        notify(args, "Evaluation queue armed: GPU 0 and GPU 1",
               f"Training wait scope: {args.wait_scope}. {len(report['runs'])} runs currently eligible; "
               f"priority GPU: {args.priority_gpu}; refresh: {args.refresh_manifest}.\nLogs: {args.queue_dir}")
        print(json.dumps(active, indent=2))
        return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "launch", "worker", "status"))
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    parser.add_argument("--results-root", type=Path, default=os.environ.get("RESULTS_ROOT", PROJECT_ROOT / "results"))
    parser.add_argument("--queue-dir", type=Path, default=os.environ.get("EVAL_QUEUE_DIR"))
    parser.add_argument("--ntfy-url", default=os.environ.get("NTFY_URL", "https://ntfy.sh/soeon_server10"))
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--quiet-seconds", type=float, default=30)
    parser.add_argument("--completed-since", help="Include runs completed on/after this YYYY-MM-DD date in KST.")
    parser.add_argument("--priority-runs-file", type=Path, help="One model_folder/run_name per line, evaluated first in file order.")
    parser.add_argument("--priority-gpu", type=int, choices=(0, 1), help="Pin all priority runs to this GPU.")
    parser.add_argument("--skip-evaluated", action="store_true", help="Skip final models with complete evaluations matching the requested settings.")
    parser.add_argument("--wait-scope", choices=("all", "gpu"), default="all",
                        help="Wait for all project training, or only training on this worker's GPU.")
    parser.add_argument("--refresh-manifest", action="store_true",
                        help="Append newly completed runs until the current training queues finish.")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.quiet_seconds < 0:
        parser.error("poll-seconds must be positive and quiet-seconds nonnegative")
    args.results_root = args.results_root.expanduser().resolve()
    if not args.results_root.is_dir():
        parser.error(f"Results root does not exist: {args.results_root}")
    if args.queue_dir is None:
        current = QUEUE_ROOT / "current.json"
        if args.mode == "launch":
            args.queue_dir = QUEUE_ROOT / datetime.now().strftime("queue-%Y%m%d_%H%M%S")
        elif current.exists():
            args.queue_dir = Path(read_json(current)["queue_dir"])
        else:
            args.queue_dir = QUEUE_ROOT / "manual"
    args.queue_dir = args.queue_dir.expanduser().resolve()
    args.priority_runs = []
    if args.priority_runs_file:
        for line in args.priority_runs_file.read_text().splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            run = (args.results_root / value).resolve()
            if not run.is_relative_to(args.results_root) or not run.is_dir():
                parser.error(f"Priority run does not exist under results: {value}")
            relative = str(run.relative_to(args.results_root))
            if relative not in args.priority_runs:
                args.priority_runs.append(relative)
    # Detached and resumed workers retain the launch-time selection/settings.
    config_path = args.queue_dir / "queue_config.json"
    if args.mode == "worker" and config_path.exists():
        config = read_json(config_path)
        for key, value in config.get("selection", {}).items():
            setattr(args, key, value)
        for key, value in config.get("scheduling", {}).items():
            setattr(args, key, value)
        args.settings = config.get("settings")
    if args.wait_scope == "gpu":
        args.refresh_manifest = True
    if args.priority_gpu is not None and not args.priority_runs:
        parser.error("priority-gpu requires a nonempty priority-runs-file")
    if args.completed_since:
        try:
            date.fromisoformat(args.completed_since)
        except ValueError:
            parser.error("completed-since must be a YYYY-MM-DD date")
    return args


def handle_termination(signum, frame):
    raise KeyboardInterrupt("SIGTERM")


def main():
    args = parse_args()
    if args.dry_run or args.mode == "plan":
        print_plan(selected_runs(args), args.gpu)
        print(f"\nLive training processes: {training_processes()}")
        return 0
    if args.mode == "status":
        print(f"Queue: {args.queue_dir}")
        for gpu in (0, 1):
            path = args.queue_dir / f"gpu{gpu}_state.json"
            print(json.dumps(read_json(path), indent=2) if path.exists() else f"GPU {gpu}: no state yet")
        return 0
    if args.mode == "launch":
        return launch(args)
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, handle_termination)
    return worker(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BlockingIOError:
        sys.exit("An evaluation worker already holds this GPU/launcher lock.")
