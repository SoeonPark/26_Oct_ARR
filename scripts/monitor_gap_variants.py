"""CPU-only diagnostics from completed validation embedding dumps, never models.

Run once by default; --watch polls for new dumps. Native top-1 uses the actual
shell score and splits credit across exact ties. It is a sampled batch metric,
not full retrieval. Positive radial projection favors contraction in embedding
space; it does not predict the optimizer's parameter-space update.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ("en-ko", "en-ja", "en-es")
LOSSES = ("gap_distance_infonce", "gap_distance_detach", "gap_distance_rms")
RMS_FLOOR_SQUARED = 1e-12
CACHE_VERSION = 2


def read_json(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def logits_and_loss(d, loss, temperature=.05, scale=1.):
    radius = np.diag(d).mean()
    q = np.sqrt(max(float(np.mean(d*d)), RMS_FLOOR_SQUARED)) if loss == "gap_distance_rms" else 1.
    z = -np.square((d-radius)/(scale*q))/temperature
    def probabilities(a):
        shifted = a-a.max(axis=1, keepdims=True)
        logp = shifted-np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return np.exp(logp), logp
    row_p, row_logp = probabilities(z)
    col_p, col_logp = probabilities(z.T)
    selected_loss = -.5*(np.diag(row_logp).mean()+np.diag(col_logp).mean())
    gradient_logits = (.5*(row_p+col_p.T)-np.eye(len(d)))/len(d)
    return z, float(selected_loss), gradient_logits


def radial_diagnostics(d, loss, temperature=.05, scale=1.):
    z, value, g = logits_and_loss(d, loss, temperature, scale)
    floor = float(np.mean(d*d)) < RMS_FLOOR_SQUARED
    if loss == "gap_distance_detach":
        backward = float(np.sum(g*(-2*(d-np.diag(d).mean())*d/(scale*scale*temperature))))
    elif loss == "gap_distance_rms" and not floor:
        backward = 0.  # Live radius AND live RMS scale cancel uniform scaling.
    else:
        backward = float(2*np.sum(g*z))
    eps = 1e-5
    forward = (logits_and_loss(d*np.exp(eps), loss, temperature, scale)[1]
               - logits_and_loss(d*np.exp(-eps), loss, temperature, scale)[1])/(2*eps)
    return value, backward, float(forward)


def tie_aware_top1(z):
    credits, tie_counts = [], []
    for a in (z, z.T):
        winners = a == a.max(axis=1, keepdims=True)
        counts = winners.sum(axis=1)
        credits.extend((np.diag(winners)/counts).tolist())
        tie_counts.extend(counts.tolist())
    return float(np.mean(credits)), float(np.mean(np.array(tie_counts)>1)), int(max(tie_counts))


def batch_metrics(x, y, loss, temperature=.05, scale=1.):
    if loss not in LOSSES or x.shape != y.shape or x.ndim != 2 or len(x) < 2:
        raise ValueError("Expected corresponding B>=2 embeddings and a supported Gap objective")
    if not np.isfinite([temperature, scale]).all() or temperature <= 0 or scale <= 0:
        raise ValueError("Temperature and scale must be finite and positive")
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    count = int((~np.isfinite(x)).sum()+(~np.isfinite(y)).sum())
    if count:
        return {"nonfinite_elements": count}
    d = np.linalg.norm(x[:, None, :]-y[None, :, :], axis=-1)
    positive = np.diag(d)
    off = ~np.eye(len(d), dtype=bool)
    q2 = float(np.mean(d*d))
    h = np.concatenate((x, y))
    centered = float(np.sqrt(np.mean(np.sum((h-h.mean(axis=0))**2, axis=1))))
    norm_rms = float(np.sqrt(np.mean(np.sum(h*h, axis=1))))
    residual2 = (d-positive.mean())**2
    row_wrong = np.where(off, residual2, np.inf).min(axis=1)
    col_wrong = np.where(off, residual2, np.inf).min(axis=0)
    shell_margin = np.concatenate((row_wrong-np.diag(residual2), col_wrong-np.diag(residual2))) / max(q2, RMS_FLOOR_SQUARED)
    z, selected, _ = logits_and_loss(d, loss, temperature, scale)
    top1, ties, max_tie = tie_aware_top1(z)
    result = dict(nonfinite_elements=0, raw_positive_mean=float(positive.mean()),
                  raw_positive_std=float(positive.std()), raw_negative_mean=float(d[off].mean()),
                  all_pair_distance_rms=float(np.sqrt(q2)), centered_embedding_rms=centered,
                  embedding_norm_rms=norm_rms, spread_to_norm_ratio=centered/norm_rms if norm_rms else 0.,
                  native_batch_top1_pct=100*top1, tied_query_fraction=ties, maximum_tie_count=max_tie,
                  normalized_shell_margin=float(shell_margin.mean()), selected_loss=selected,
                  rms_floor_active=q2 <= RMS_FLOOR_SQUARED)
    for mode, prefix in zip(LOSSES, ("original", "detach", "rms")):
        value, backward, forward = radial_diagnostics(d, mode, temperature, scale)
        result.update({f"{prefix}_diagnostic_loss": value,
                       f"{prefix}_backward_radial": backward,
                       f"{prefix}_reforward_radial": forward})
    result["selected_backward_radial"] = result[{LOSSES[0]: "original", LOSSES[1]: "detach", LOSSES[2]: "rms"}[loss]+"_backward_radial"]
    result["selected_reforward_radial"] = result[{LOSSES[0]: "original", LOSSES[1]: "detach", LOSSES[2]: "rms"}[loss]+"_reforward_radial"]
    result["selected_contraction_indicator"] = result["selected_backward_radial"] > 1e-9
    return result


def complete_batches(records, embeddings, limit=64):
    """Preserve logged candidate membership; never repack incomplete batches."""
    groups, skipped = {}, []
    for record in records[:limit]:
        groups.setdefault(record.get("batch_id"), []).append(record)
    output = []
    for batch_id, group in groups.items():
        sizes = {record.get("actual_batch_size") for record in group}
        size = next(iter(sizes)) if len(sizes) == 1 else None
        positions = [record.get("batch_position") for record in group]
        if not batch_id or not isinstance(size, int) or size < 2 or len(group) != size or set(positions) != set(range(size)):
            skipped.append({"batch_id": batch_id, "reason": "incomplete_or_invalid_batch_metadata"})
            continue
        group = sorted(group, key=lambda r: r["batch_position"])
        ids = [record["sample_id"] for record in group]
        if any(record.get("batch_sample_ids", ids) != ids for record in group):
            raise ValueError(f"Candidate identity mismatch: {batch_id}")
        x, y = (np.stack([embeddings[r["embedding_keys"][f"{side}_embedding_key"]] for r in group])
                for side in ("source", "target"))
        identity = [{key: r.get(key) for key in ("sample_id", "source_token_hash", "target_token_hash")}
                    for r in group]
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        output.append((batch_id, fingerprint, group, x, y))
    return output, skipped


def discover_runs(manifest, state, baselines):
    runs, pending = [], []
    for job in manifest["jobs"]:
        if not job.get("lane"):
            continue
        current = state.get("jobs", {}).get(job["id"], {})
        path = current.get("run")
        if not path and current.get("log"):
            path = Path(job.get("results_root", manifest["results_root"])) / job["model"].replace("/", "__") / Path(current["log"]).stem
        if not path:
            pending.append({"job_id": job["id"], "status": current.get("training_status", "queued")})
            continue
        runs.append(dict(run=str(Path(path).resolve()), job_id=job["id"], config=job["config"], baseline=False))
    for path in baselines:
        path = path.expanduser().resolve()
        config = read_json(path / "experiment_config.json")
        if (config.get("alignment_loss") != LOSSES[0] or config.get("training_seed") != 42
                or config.get("alignment_hidden_state_layer") != -1):
            raise ValueError(f"Baseline must be original Gap, seed42, final layer: {path}")
        runs.append(dict(run=str(path), job_id="baseline:"+path.name, config=config, baseline=True))
    return runs, pending


def fingerprint(paths):
    return [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["run"], row["global_step"]), []).append(row)
    output = []
    identity_keys = ("observed_at", "run", "job_id", "model", "loss", "layer", "seed", "global_step", "baseline")
    excluded = set(identity_keys) | {"pair", "batch_id", "batch_identity", "batch_size", "source_json"}
    for group in groups.values():
        row = {key: group[0][key] for key in identity_keys}
        row.update(batch_count=len(group), sampled_pair_count=sum(r["batch_size"] for r in group),
                   matching_candidate_set_id=hashlib.sha256("|".join(sorted(r["pair"]+":"+r["batch_identity"] for r in group)).encode()).hexdigest())
        for key in set().union(*(r.keys() for r in group))-excluded:
            values = [r[key] for r in group if isinstance(r.get(key), (int, float, bool))]
            if values:
                if key == "nonfinite_elements":
                    row[key] = int(sum(values))
                elif key == "maximum_tie_count":
                    row[key] = int(max(values))
                else:
                    row[key] = float(np.mean(values))
        output.append(row)
    return sorted(output, key=lambda r: (r["model"], r["job_id"], r["global_step"]))


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in rows})
    if not models:
        return
    fig, axes = plt.subplots(len(models), 6, figsize=(25, 4*len(models)), squeeze=False)
    metrics = (("centered_embedding_rms", "Raw centered RMS", "symlog"),
               ("all_pair_distance_rms", "All-pair distance RMS (q)", "symlog"),
               ("normalized_shell_margin", "Nearest-wrong shell margin / q²", "linear"),
               ("native_batch_top1_pct", "Native B16 top-1 (tie-aware, %)", "linear"),
               ("selected_backward_radial", "Backward radial projection (+ contracts)", "symlog"),
               ("downstream_validation_loss", "MASSIVE In validation loss (4-language mean)", "linear"))
    for i, model in enumerate(models):
        subset = [r for r in rows if r["model"] == model]
        for run in dict.fromkeys(r["run"] for r in subset):
            rr = [r for r in subset if r["run"] == run]
            label = {LOSSES[0]: "Original Gap", LOSSES[1]: "Gap detach", LOSSES[2]: "Gap RMS"}[rr[0]["loss"]]
            label += " (baseline)" if rr[0]["baseline"] else ""
            for col, (metric, title, scale) in enumerate(metrics):
                valid = [r for r in rr if metric in r]
                axes[i, col].plot([r["global_step"] for r in valid], [r[metric] for r in valid], marker=".", label=label)
                axes[i, col].set(title=model+"\n"+title, xlabel="Global step", yscale=scale)
                if col < 2:
                    axes[i, col].set_yscale("symlog", linthresh=1e-6)
                axes[i, col].grid(alpha=.2)
        axes[i, 0].legend(fontsize=8)
    fig.suptitle("Saved OPUS validation batches: en-ko / en-ja / en-es (64 pairs each)\n"
                 "Different models shown separately; derivatives are embedding-space diagnostics, not optimizer updates.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .92))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"monitor.{ext}", dpi=170)
    plt.close(fig)


def scan_once(state_dir, out, baselines=()):
    out.mkdir(parents=True, exist_ok=True)
    manifest, state = read_json(state_dir/"manifest.json"), read_json(state_dir/"state.json")
    runs, pending = discover_runs(manifest, state, baselines)
    cache_path = out/"cache.json"
    cache = read_json(cache_path) if cache_path.exists() else {"version": CACHE_VERSION, "steps": {}}
    if cache.get("version") != CACHE_VERSION:
        cache = {"version": CACHE_VERSION, "steps": {}}
    changed, retry = False, []
    for run in runs:
        config = run["config"]
        for path in sorted((Path(run["run"])/"eval_samples").glob("step-*.json")):
            match = re.fullmatch(r"step-(\d+)\.json", path.name)
            if not match:
                continue
            key = str(path)
            paths = [path, path.with_name(path.stem+"_embeddings.pkl")]
            metrics_path = path.with_name(path.stem+"_metrics.json")
            if metrics_path.exists():
                paths.append(metrics_path)
            try:
                before = fingerprint(paths)
                cache_key = json.dumps([before, config], sort_keys=True)
                if cache["steps"].get(key, {}).get("fingerprint") == cache_key:
                    continue
                samples = read_json(path)
                downstream = {}
                if metrics_path.exists():
                    metrics = read_json(metrics_path)
                    names = ["eval_massive_in_"+language for language in ("en", "ko", "ja", "es")]
                    if all(name in metrics for name in names):
                        downstream["downstream_validation_loss"] = float(np.mean([metrics[name]["selected_loss_mean"] for name in names]))
                # Only local, trusted training artifacts are unpickled.
                with paths[1].open("rb") as handle:
                    embeddings = pickle.load(handle)
                rows = []
                for pair in PAIRS:
                    batches, skipped = complete_batches(samples.get("alignment/"+pair, []), embeddings)
                    if skipped or sum(len(b[2]) for b in batches) != 64:
                        raise ValueError(f"Waiting for 64 complete saved pairs: {pair}; skipped={skipped}")
                    for batch_id, identity, group, x, y in batches:
                        if any(r.get("alignment_loss_type", config["alignment_loss"]) != config["alignment_loss"] for r in group):
                            raise ValueError("Logged objective differs from frozen configuration")
                        rows.append(dict(observed_at=stamp(), run=run["run"], job_id=run["job_id"],
                                         model=config["model_name"], loss=config["alignment_loss"],
                                         layer=config["alignment_hidden_state_layer"], seed=config["training_seed"],
                                         baseline=run["baseline"], global_step=int(match[1]), pair=pair,
                                         batch_id=batch_id, batch_identity=identity, batch_size=len(group),
                                         source_json=str(path), **batch_metrics(x, y, config["alignment_loss"],
                                         config.get("alignment_temperature", .05), config.get("alignment_gap_scale", 1.)), **downstream))
                if fingerprint(paths) != before:
                    raise ValueError("Files changed while being read")
                cache["steps"][key] = dict(fingerprint=cache_key, rows=rows)
                changed = True
            except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as error:
                retry.append({"path": key, "reason": str(error)})
    allowed = {r["run"] for r in runs}
    rows = [r for item in cache["steps"].values() for r in item["rows"] if r["run"] in allowed]
    trajectory = summarize(rows)
    latest = list({r["run"]: r for r in trajectory}.values())
    panel_matches = {model: len({row["matching_candidate_set_id"] for row in trajectory if row["model"] == model}) <= 1
                     for model in {row["model"] for row in trajectory}}
    status = dict(updated_at=stamp(), completed_step_count=len(trajectory), pending_jobs=pending,
                  retry=retry, runs=runs, queue_status=state.get("status"),
                  completed_steps_share_sample_and_token_ids_by_model=panel_matches,
                  nonfinite_elements_latest=sum(r.get("nonfinite_elements", 0) for r in latest),
                  definition="B16 complete sampled validation batches; native shell ranking, not full retrieval. Positive backward radial means contraction preference. Detach backward differs from reforward derivative. RMS invariance holds only above its floor.",
                  latest=latest)
    previous = read_json(out/"status.json") if (out/"status.json").exists() else None
    status_changed = previous is None or any(previous.get(k) != status[k] for k in ("pending_jobs", "retry", "runs", "queue_status"))
    status["last_polled_at"] = stamp()
    status["monitor_pid"] = os.getpid()
    if not changed and not status_changed:
        status["updated_at"] = previous["updated_at"]
    if changed or status_changed:
        atomic_json(cache_path, cache)
        write_csv(out/"per_batch.csv", rows)
        write_csv(out/"trajectory.csv", trajectory)
        write_csv(out/"latest.csv", latest)
        print(json.dumps({"updated_at": status["updated_at"], "completed_steps": len(trajectory),
                          "pending_jobs": len(pending), "retry_files": len(retry)}, ensure_ascii=False), flush=True)
    if trajectory and (changed or any(not (out/f"monitor.{ext}").exists() for ext in ("png", "pdf"))):
        plot(trajectory, out)
    atomic_json(out/"status.json", status)
    return status, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=ROOT/"logs/gap_variants_priority_20261010")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"logs/gap_variants_priority_20261010/monitor")
    parser.add_argument("--baseline", type=Path, action="append", default=[])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--once", action="store_true", help="Default: inspect currently available completed steps")
    modes.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=120.)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    while True:
        try:
            status, _ = scan_once(args.state_dir.resolve(), args.output_dir.resolve(), args.baseline)
            if args.watch and status.get("queue_status") in ("completed", "failed"):
                print(json.dumps({"updated_at": stamp(), "monitor_exit": "queue_"+status["queue_status"]}), flush=True)
                break
        except (OSError, ValueError) as error:
            if not args.watch:
                raise
            print(json.dumps({"updated_at": stamp(), "retry_state_read": str(error)}), flush=True)
        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
