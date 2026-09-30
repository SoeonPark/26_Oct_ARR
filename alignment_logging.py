"""Detached distance diagnostics shared by training and checkpoint evaluation."""

import hashlib
import json
from pathlib import Path

import torch


SCHEMA_VERSION = 1
_FIELDS = ("gap_distance", "per_sample_loss", "source_norm", "target_norm", "positive_cosine")


def _values(outputs):
    # One device transfer per observation instead of synchronizing per field.
    values = torch.stack([outputs[key].detach() for key in _FIELDS])
    return dict(zip(_FIELDS, values.to(device="cpu", dtype=torch.float64).unbind(0)))


def _pair_indices(batch):
    groups = {}
    for index, pair in enumerate(batch["lang_pair"]):
        groups.setdefault(pair, []).append(index)
    return groups


class AlignmentStatistics:
    """All-example statistics, independent of detailed sample log limits.

    Training keeps constant-size sums per pair. Fixed-model evaluation also
    keeps scalar distances (not embeddings) for variance and exact quantiles.
    """

    def __init__(self, retain_distances=True):
        self.retain_distances = retain_distances
        self.groups = {}

    def update(self, batch, outputs):
        values = _values(outputs)
        for pair, indices in _pair_indices(batch).items():
            state = self.groups.setdefault(pair, {
                "count": 0, "sums": dict.fromkeys(_FIELDS, 0.0),
                "batch_variance_sum": 0.0, "distances": [],
            })
            distances = values["gap_distance"][indices]
            state["count"] += len(indices)
            for key in _FIELDS:
                state["sums"][key] += values[key][indices].sum().item()
            state["batch_variance_sum"] += (
                values["per_sample_loss"][indices].sum().item()
                if outputs["alignment_loss_type"] == "gap_consistency"
                else (distances - distances.mean()).square().sum().item()
            )
            if self.retain_distances:
                state["distances"].append(distances.clone())

    def summary(self, corpus=True):
        if corpus and not self.retain_distances:
            raise ValueError("Corpus statistics require retained evaluation distances.")
        result = {}
        for pair, state in self.groups.items():
            count = state["count"]
            sums = state["sums"]
            metrics = {
                "num_examples": count,
                "selected_loss_mean": sums["per_sample_loss"] / count,
                "batch_gap_loss_mean": state["batch_variance_sum"] / count,
                "gap_distance_mean": sums["gap_distance"] / count,
                "source_norm_mean": sums["source_norm"] / count,
                "target_norm_mean": sums["target_norm"] / count,
                "positive_cosine_mean": sums["positive_cosine"] / count,
            }
            if corpus:
                distances = torch.cat(state["distances"])
                variance = distances.var(unbiased=False).item()
                quantiles = torch.quantile(
                    distances, torch.tensor([0.05, 0.5, 0.95], dtype=distances.dtype),
                ).tolist()
                metrics.update({
                    "corpus_gap_distance_mean": distances.mean().item(),
                    "corpus_gap_distance_variance": variance,
                    "corpus_gap_distance_std": variance ** 0.5,
                    "gap_distance_min": distances.min().item(),
                    "gap_distance_max": distances.max().item(),
                    **dict(zip(("gap_distance_p05", "gap_distance_p50", "gap_distance_p95"), quantiles)),
                })
            result[pair] = metrics
        return result


def batch_record(batch, outputs, context):
    distances = outputs["gap_distance"].detach().double().cpu()
    return {
        **context,
        "schema_version": SCHEMA_VERSION,
        "alignment_loss_type": outputs["alignment_loss_type"],
        "actual_batch_size": len(batch["lang_pair"]),
        "sample_ids": list(batch["sample_id"]),
        "language_pairs": list(batch["lang_pair"]),
        "batch_loss": outputs["loss"].detach().item(),
        "pair_mean_distances": {
            pair: (outputs["gap_distance_mean"].detach().item()
                   if outputs["alignment_loss_type"] == "gap_consistency"
                   else distances[indices].mean().item())
            for pair, indices in _pair_indices(batch).items()
        },
    }


def _token_metadata(batch, side, index):
    mask = batch[f"{side}_attention_mask"][index].detach().bool()
    token_ids = batch[f"{side}_input_ids"][index].detach()[mask].cpu().tolist()
    encoded = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return {
        f"{side}_num_tokens": len(token_ids),
        f"{side}_token_hash": hashlib.sha256(encoded).hexdigest(),
    }


def alignment_records(batch, outputs, indices, context):
    """Describe actual forward observations; never run the model or sample RNG."""
    indices = list(indices)
    if not indices:
        return []
    values = _values(outputs)
    means = {pair: (outputs["gap_distance_mean"].detach().item()
                   if outputs["alignment_loss_type"] == "gap_consistency"
                   else values["gap_distance"][positions].mean().item())
             for pair, positions in _pair_indices(batch).items()}
    records = []
    for index in indices:
        pair = batch["lang_pair"][index]
        distance = values["gap_distance"][index].item()
        residual = distance - means[pair]
        records.append({
            **context,
            "schema_version": SCHEMA_VERSION,
            "record_id": f"{context['batch_id']}/sample-{index}",
            "sample_id": batch["sample_id"][index],
            "batch_position": index,
            "actual_batch_size": len(batch["lang_pair"]),
            "lang_pair": pair,
            "alignment_loss_type": outputs["alignment_loss_type"],
            "source_text": batch["source_text"][index],
            "target_text": batch["target_text"][index],
            **{key: values[key][index].item() for key in _FIELDS},
            "batch_gap_distance_mean": means[pair],
            "gap_residual": residual,
            "gap_squared_residual": (
                values["per_sample_loss"][index].item()
                if outputs["alignment_loss_type"] == "gap_consistency" else residual ** 2
            ),
            **_token_metadata(batch, "source", index),
            **_token_metadata(batch, "target", index),
        })
    return records


def append_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
