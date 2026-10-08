import json
import os
from functools import partial
from pathlib import Path
from uuid import uuid4

import torch
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import seed_worker
from transformers.utils import logging
import pickle

from alignment_logging import (
    AlignmentStatistics, alignment_records, append_jsonl, batch_record, write_json,
)
from config import PAIR_BATCH_LOSSES, resolve_alignment_loss
from samplers import AlignmentEvalBatchSampler, PairBatchSampler


logger = logging.get_logger(__name__)


class AlternativeRoutingTrainer(Trainer):

    def __init__(
        self,
        *args,
        schedule=("alignment", "downstream"),
        eval_forward_type="downstream",
        training_type="alternative",
        total_steps=None,
        eval_sample_log_limit=64,
        alignment_batching="mixed",
        **kwargs,
    ):
        self.schedule = schedule
        self.eval_forward_type = eval_forward_type
        self.training_type = training_type
        self.total_steps = total_steps
        if alignment_batching not in {"mixed", "same_pair"}:
            raise ValueError(f"Unknown alignment_batching: {alignment_batching}")
        self.alignment_batching = alignment_batching

        # Per-sample validation records, keyed by "<objective>/<language>".
        # Filled in prediction_step and written out once per evaluation round.
        self.eval_sample_log_limit = eval_sample_log_limit
        self.eval_sample_buffer = {}
        self.eval_sample_embedding_buffer = {}

        # Accumulate objective-specific losses between Trainer log events.
        self._objective_loss_sums = {
            "alignment": None,
            "downstream": None,
        }
        self._objective_loss_counts = {
            "alignment": 0,
            "downstream": 0,
        }

        super().__init__(*args, **kwargs)

        self.model_accepts_loss_kwargs = False
        config = getattr(self.model, "experiment_config", None)
        self.downstream_micro_batch_size = getattr(config, "downstream_micro_batch_size", 0)
        self._downstream_loss_weight = None
        if self.downstream_micro_batch_size < 0:
            raise ValueError("downstream_micro_batch_size must be nonnegative.")
        if self.downstream_micro_batch_size and (self.accelerator.num_processes != 1 or self.args.n_gpu > 1):
            raise ValueError("SFT microbatch accumulation currently requires one process/GPU.")
        self.alignment_loss = resolve_alignment_loss(config)
        self.train_sample_log_interval = getattr(config, "train_sample_log_interval", 1000)
        self.train_sample_log_limit = getattr(config, "train_sample_log_limit", 8)
        if min(self.train_sample_log_interval, self.train_sample_log_limit, self.eval_sample_log_limit) < 0:
            raise ValueError("Sample logging intervals and limits must be nonnegative.")
        if self.alignment_loss in PAIR_BATCH_LOSSES:
            if self.accelerator.num_processes != 1 or self.args.n_gpu > 1:
                raise ValueError("Pair-based alignment requires one process and one visible GPU.")
            if self.training_type != "transfer_only" and (
                self.alignment_batching != "same_pair" or self.args.train_batch_size < 2
            ):
                raise ValueError("Gap training requires same_pair and batch_size >= 2.")
        self._gap_eval_dataloaders = {}
        self._logging_session = uuid4().hex
        self._train_alignment_statistics = AlignmentStatistics(retain_distances=False)
        self._train_observed_step = None
        self._train_microbatch_index = 0
        self._eval_round = 0
        self._evaluation_depth = 0
        self._eval_context = None
        self._eval_alignment_statistics = None
        self._eval_weighted_loss = None
        self._eval_batch_buffer = []
        self._eval_metric_buffer = {}

    def objective_at(self, step):
        """Shared schedule for the live Trainer and planned sampler steps."""
        if self.training_type == "transfer_only":
            return "downstream"
        elif self.training_type == "contrastive_only":
            return "alignment"
        elif self.training_type == "contrastive_then_transfer":
            half_steps = self.total_steps // 2
            if step < half_steps:
                return "alignment"
            else:
                return "downstream"
        elif self.training_type == "alternative":
            return self.schedule[
                step
                % len(self.schedule)
            ]
        else:
            raise ValueError(
                f"Unknown training_type: {self.training_type}"
            )

    def objective_for_step(self):
        return self.objective_at(self.state.global_step)

    def get_train_dataloader(self):
        dataset = getattr(self, "train_dataset", None)
        downstream_dataset = getattr(dataset, "downstream_dataset", None)
        downstream_ranges = getattr(downstream_dataset, "language_ranges", None)
        downstream_sampling = getattr(downstream_dataset, "downstream_sampling", "language_balanced")
        if downstream_sampling == "balanced_mixed":
            downstream_ranges = downstream_dataset.balanced_language_ranges
        if self.alignment_batching == "mixed" and downstream_ranges is None:
            return super().get_train_dataloader()

        if self.accelerator.num_processes != 1 or self.args.n_gpu > 1:
            raise ValueError(
                "same_pair currently requires one process and one visible GPU."
            )
        if self.total_steps != self.args.max_steps:
            raise ValueError("same_pair requires total_steps == max_steps > 0.")

        if downstream_ranges and self.training_type == "alternative":
            if self.args.max_steps % 2 or self.schedule != ("alignment", "downstream"):
                raise ValueError("WMT requires an even step budget and exact alignment/downstream alternation.")
        sampler = PairBatchSampler(
            pair_ranges=(dataset.alignment_dataset.pair_ranges
                         if self.alignment_batching == "same_pair"
                         else {"mixed": (0, len(dataset.alignment_dataset))}),
            downstream_size=len(dataset.downstream_dataset),
            batch_size=self._train_batch_size,
            num_steps=self.args.max_steps,
            accumulation_steps=self.args.gradient_accumulation_steps,
            seed=(
                self.args.data_seed
                if self.args.data_seed is not None else self.args.seed
            ),
            objective_at=self.objective_at,
            downstream_ranges=(
                None if downstream_sampling == "proportional"
                else downstream_ranges
            ),
            downstream_sampling=downstream_sampling,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            multiprocessing_context=self.args.dataloader_multiprocessing_context,
            worker_init_fn=partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
            # Keep DataLoader's ordered-delivery default: worker completion
            # order must not change the objective schedule.
        )
        return self.accelerator.prepare(loader)

    def get_eval_dataloader(self, eval_dataset=None):
        dataset = (
            self.eval_dataset[eval_dataset] if isinstance(eval_dataset, str)
            else eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        alignment_dataset = getattr(dataset, "alignment_dataset", None)
        if self.alignment_loss not in PAIR_BATCH_LOSSES or alignment_dataset is None:
            return super().get_eval_dataloader(eval_dataset)
        if getattr(dataset, "downstream_dataset", None) is not None:
            raise ValueError("Gap validation requires an alignment-only dataset.")
        key = id(dataset)
        if self.args.dataloader_persistent_workers and key in self._gap_eval_dataloaders:
            return self._gap_eval_dataloaders[key]
        loader = DataLoader(
            dataset,
            batch_sampler=AlignmentEvalBatchSampler(
                alignment_dataset.pair_ranges, self.args.eval_batch_size,
            ),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            multiprocessing_context=self.args.dataloader_multiprocessing_context,
        )
        loader = self.accelerator.prepare(loader)
        if self.args.dataloader_persistent_workers:
            self._gap_eval_dataloaders[key] = loader
        return loader

    def objective_update_counts(self, completed_steps):
        if self.training_type == "transfer_only":
            return 0, completed_steps

        if self.training_type == "contrastive_only":
            return completed_steps, 0

        if self.training_type == "contrastive_then_transfer":
            half_steps = self.total_steps // 2
            alignment_steps = min(completed_steps, half_steps)
            downstream_steps = max(0, completed_steps - half_steps)
            return alignment_steps, downstream_steps

        if self.training_type == "alternative":
            schedule_length = len(self.schedule)
            full_cycles, remainder = divmod(
                completed_steps,
                schedule_length,
            )
            alignment_steps = full_cycles * self.schedule.count("alignment")
            downstream_steps = full_cycles * self.schedule.count("downstream")

            for objective in self.schedule[:remainder]:
                if objective == "alignment":
                    alignment_steps += 1
                elif objective == "downstream":
                    downstream_steps += 1

            return alignment_steps, downstream_steps

        raise ValueError(f"Unknown training_type: {self.training_type}")

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch=None,
    ):
        if isinstance(model, torch.nn.DataParallel):
            raise RuntimeError(
                "4-bit bitsandbytes models must not be trained with "
                "torch.nn.DataParallel. Use one visible GPU or DDP."
            )

        objective = self.objective_for_step()
        inputs = dict(inputs)

        if self.alignment_batching == "same_pair":
            present = {
                name for name in ("alignment", "downstream")
                if name in inputs
            }
            if present != {objective}:
                raise RuntimeError(
                    f"Step {self.state.global_step}: "
                    f"expected {objective}, got {sorted(present)}"
                )
            if objective == "alignment":
                pairs = set(inputs["alignment"]["lang_pair"])
                if len(pairs) != 1:
                    raise RuntimeError(
                        f"Alignment batch must contain one language pair: {pairs}"
                    )

        inputs["forward_type"] = objective

        micro_size = self.downstream_micro_batch_size
        if objective == "downstream" and micro_size and micro_size < inputs["downstream"]["input_ids"].size(0):
            data = inputs["downstream"]
            batch_size = data["input_ids"].size(0)
            total_tokens = (data["labels"][:, 1:] != -100).sum().item()
            if not total_tokens:
                raise ValueError("SFT batch has no supervised next-token labels.")
            losses = []
            try:
                for start in range(0, batch_size, micro_size):
                    micro = {
                        key: value[start:start + micro_size]
                        if ((torch.is_tensor(value) and value.ndim and value.size(0) == batch_size)
                            or (isinstance(value, (list, tuple)) and len(value) == batch_size))
                        else value
                        for key, value in data.items()
                    }
                    num_tokens = (micro["labels"][:, 1:] != -100).sum().item()
                    if not num_tokens:
                        continue
                    # HF CE is a token mean. Weight by supervised token counts,
                    # not equally by microbatches with different target lengths.
                    self._downstream_loss_weight = num_tokens / total_tokens
                    losses.append(super().training_step(
                        model, {**inputs, "downstream": micro}, num_items_in_batch,
                    ))
            finally:
                self._downstream_loss_weight = None
            loss = sum(losses)
        else:
            loss = super().training_step(model, inputs, num_items_in_batch)

        # Trainer divides the returned loss by gradient accumulation steps.
        # Undo that scaling so logged losses remain comparable.
        accumulation_steps = getattr(
            self,
            "current_gradient_accumulation_steps",
            self.args.gradient_accumulation_steps,
        )
        loss_for_logging = (
            loss.detach().float().mean()
            * accumulation_steps
        )

        if self._objective_loss_sums[objective] is None:
            self._objective_loss_sums[objective] = loss_for_logging
        else:
            self._objective_loss_sums[objective] += loss_for_logging
        self._objective_loss_counts[objective] += 1

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        observe = model.training and inputs.get("forward_type") == "alignment"
        if observe:
            inputs["return_per_sample"] = True
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch,
        )
        # CustomModel supplies these diagnostics; simple routing-only test or
        # third-party models can still return just their scalar loss.
        if observe and "gap_distance" in outputs:
            self._record_alignment_training(inputs["alignment"], outputs)
        if self._downstream_loss_weight is not None and inputs.get("forward_type") == "downstream":
            loss = loss * self._downstream_loss_weight
        return (loss, outputs) if return_outputs else loss

    def _observation_context(self, phase):
        config = getattr(self.model, "experiment_config", None)
        return {
            "run_id": self.args.run_name or str(self.args.output_dir),
            "session_id": self._logging_session,
            "rank": self.args.process_index,
            "phase": phase,
            "global_step": self.state.global_step,
            "alignment_hidden_state_layer": getattr(config, "alignment_hidden_state_layer", None),
            "alignment_hidden_state_position": getattr(config, "alignment_hidden_state_position", None),
        }

    def _record_alignment_training(self, batch, outputs):
        self._train_alignment_statistics.update(batch, outputs)
        if self._train_observed_step != self.state.global_step:
            self._train_observed_step = self.state.global_step
            self._train_microbatch_index = 0
        self._train_microbatch_index += 1
        update_index = self.objective_update_counts(self.state.global_step)[0] + 1
        interval = self.train_sample_log_interval
        if not interval or not self.train_sample_log_limit or update_index % interval:
            return
        context = {
            **self._observation_context("train"),
            "split": "train", "scope": "in",
            "global_step_before_update": self.state.global_step,
            "alignment_update_index": update_index,
            "microbatch_index": self._train_microbatch_index,
            "dropout_active": bool(self.model.training),
            "batch_id": (
                f"{self._logging_session}/rank-{self.args.process_index}/train/"
                f"step-{self.state.global_step}/microbatch-{self._train_microbatch_index}"
            ),
        }
        indices = range(min(self.train_sample_log_limit, len(batch["lang_pair"])))
        records = alignment_records(batch, outputs, indices, context)
        directory = Path(self.args.output_dir) / "train_samples"
        suffix = f"-rank{self.args.process_index}" if self.accelerator.num_processes > 1 else ""
        append_jsonl(directory / f"alignment-observations{suffix}.jsonl", records)
        append_jsonl(directory / f"alignment-batches{suffix}.jsonl", [batch_record(batch, outputs, context)])

    def _gather_objective_logs(self):
        zero = torch.zeros(
            (),
            device=self.args.device,
            dtype=torch.float32,
        )
        alignment_sum = self._objective_loss_sums["alignment"]
        downstream_sum = self._objective_loss_sums["downstream"]

        if alignment_sum is None:
            alignment_sum = zero
        if downstream_sum is None:
            downstream_sum = zero

        local_statistics = torch.stack(
            [
                alignment_sum,
                torch.tensor(
                    float(self._objective_loss_counts["alignment"]),
                    device=self.args.device,
                ),
                downstream_sum,
                torch.tensor(
                    float(self._objective_loss_counts["downstream"]),
                    device=self.args.device,
                ),
            ]
        ).unsqueeze(0)

        gathered_statistics = self.accelerator.gather(local_statistics)
        statistics = gathered_statistics.sum(dim=0)

        objective_logs = {}
        alignment_count = statistics[1].item()
        downstream_count = statistics[3].item()

        if alignment_count > 0:
            objective_logs["alignment_loss"] = (
                statistics[0].item() / alignment_count
            )
        if downstream_count > 0:
            objective_logs["downstream_loss"] = (
                statistics[2].item() / downstream_count
            )

        for objective in self._objective_loss_sums:
            self._objective_loss_sums[objective] = None
            self._objective_loss_counts[objective] = 0

        return objective_logs

    def log(self, logs, start_time=None):
        logs = dict(logs)

        if "loss" in logs or "train_loss" in logs:
            logs.update(self._gather_objective_logs())
            prefix = "alignment" if self.accelerator.num_processes == 1 else f"alignment_rank{self.args.process_index}_local"
            for pair, metrics in self._train_alignment_statistics.summary(corpus=False).items():
                logs.update({f"{prefix}/{pair}/{name}": value for name, value in metrics.items()})
            self._train_alignment_statistics = AlignmentStatistics(retain_distances=False)
            alignment_updates, downstream_updates = (
                self.objective_update_counts(self.state.global_step)
            )
            logs["alignment_update_count"] = alignment_updates
            logs["downstream_update_count"] = downstream_updates

        return super().log(logs, start_time)

    def _get_custom_model(self, model=None):
        candidate = self.accelerator.unwrap_model(
            model if model is not None else self.model,
            keep_torch_compile=False,
        )
        if not hasattr(candidate, "basemodel"):
            raise TypeError(
                "AlternativeRoutingTrainer expects a model with a "
                "PEFT basemodel attribute."
            )
        return candidate

    def _save(self, output_dir=None, state_dict=None):
        """Save the PEFT adapter instead of duplicating 4-bit base weights."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving PEFT checkpoint to {output_dir}")

        custom_model = self._get_custom_model()
        custom_model.basemodel.save_pretrained(
            output_dir,
            safe_serialization=True,
        )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        torch.save(
            self.args,
            os.path.join(output_dir, TRAINING_ARGS_NAME),
        )

        experiment_config = vars(
            custom_model.experiment_config
        ).copy()
        experiment_config["checkpoint_global_step"] = (
            self.state.global_step
        )
        with open(
            os.path.join(output_dir, "experiment_config.json"),
            "w",
            encoding="utf-8",
        ) as config_file:
            json.dump(
                experiment_config,
                config_file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore adapter weights; Trainer restores optimizer and RNG state."""
        config_path = Path(resume_from_checkpoint) / "experiment_config.json"
        saved_config = {}
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as config_file:
                saved_config = json.load(config_file)
        saved_loss = resolve_alignment_loss(saved_config)
        if saved_loss != self.alignment_loss:
            raise ValueError(
                f"Cannot resume {saved_loss} with alignment_loss={self.alignment_loss}."
            )
        safe_adapter_path = os.path.join(
            resume_from_checkpoint,
            "adapter_model.safetensors",
        )
        adapter_path = os.path.join(
            resume_from_checkpoint,
            "adapter_model.bin",
        )

        if not (
            os.path.isfile(safe_adapter_path)
            or os.path.isfile(adapter_path)
        ):
            return super()._load_from_checkpoint(
                resume_from_checkpoint,
                model,
            )

        custom_model = self._get_custom_model(model)
        peft_model = custom_model.basemodel

        if os.path.isfile(safe_adapter_path):
            adapter_state = load_safetensors(
                safe_adapter_path,
                device="cpu",
            )
        else:
            adapter_state = torch.load(
                adapter_path,
                map_location="cpu",
                weights_only=True,
            )

        active_adapters = getattr(
            peft_model,
            "active_adapters",
            ["default"],
        )
        if isinstance(active_adapters, str):
            adapter_name = active_adapters
        else:
            adapter_name = active_adapters[0]

        set_peft_model_state_dict(
            peft_model,
            adapter_state,
            adapter_name=adapter_name,
        )
        logger.info(
            "Loaded PEFT adapter from %s",
            resume_from_checkpoint,
        )

    def evaluation_loop(
        self, dataloader, description, prediction_loss_only=None,
        ignore_keys=None, metric_key_prefix="eval",
    ):
        self._eval_round += 1
        self._eval_alignment_statistics = AlignmentStatistics()
        self._eval_weighted_loss = [0.0, 0]
        self._eval_context = {
            **self._observation_context("validation"),
            "split": "validation",
            "scope": "out" if "_out_" in metric_key_prefix else "in" if "_in_" in metric_key_prefix else "unspecified",
            "dataset_key": metric_key_prefix,
            "evaluation_round": self._eval_round,
        }
        self._eval_batch_index = 0
        try:
            result = super().evaluation_loop(
                dataloader, description, prediction_loss_only,
                ignore_keys, metric_key_prefix,
            )
            weighted_sum, count = self._eval_weighted_loss
            # Trainer repeats scalar losses by its configured B. Our final
            # single-pair batch may have B+1, so use actual sample counts.
            if count and self.accelerator.num_processes == 1:
                result.metrics[f"{metric_key_prefix}_loss"] = weighted_sum / count
            summary = self._eval_alignment_statistics.summary()
            local_suffix = "" if self.accelerator.num_processes == 1 else f"_rank{self.args.process_index}_local"
            for pair, values in summary.items():
                pair_suffix = f"_{pair}" if len(summary) > 1 else ""
                result.metrics.update({
                    f"{metric_key_prefix}{pair_suffix}{local_suffix}_{key}": value
                    for key, value in values.items()
                })
            self._eval_metric_buffer[metric_key_prefix] = {
                **self._eval_context,
                "alignment_loss_type": self.alignment_loss,
                "aggregation_scope": "dataset" if self.accelerator.num_processes == 1 else "process_local",
                "observed_num_examples": count,
                "selected_loss_mean": weighted_sum / count if count else None,
                "distance_statistics": summary,
            }
            return result
        finally:
            self._eval_context = None
            self._eval_alignment_statistics = None
            self._eval_weighted_loss = None

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """Evaluate one objective per batch and return only its loss.

        The parent implementation cannot be reused here: `CustomModel.forward`
        takes labels through **inputs, so `find_labels` reports no label columns
        and `can_return_loss` is False, which sends the parent down its
        `loss = None` branch. Returning logits is also not an option, because
        the model outputs carry string metadata and a [B, T, vocab] tensor.
        """
        inputs = self._prepare_inputs(inputs)

        has_alignment = "alignment" in inputs
        has_downstream = "downstream" in inputs

        if has_alignment == has_downstream:
            raise ValueError(
                "Validation batches must carry exactly one objective, because "
                "a single eval_loss cannot describe two of them. Got keys: "
                f"{sorted(inputs.keys())}"
            )

        forward_type = "alignment" if has_alignment else "downstream"

        with torch.no_grad():
            outputs = model(
                forward_type=forward_type,
                return_per_sample=True,
                **inputs,
            )

        batch = inputs[forward_type]
        if self._eval_weighted_loss is not None:
            key = "source_input_ids" if has_alignment else "input_ids"
            count = batch[key].size(0)
            self._eval_weighted_loss[0] += outputs["loss"].detach().double().item() * count
            self._eval_weighted_loss[1] += count
            self._eval_batch_index += 1
            self._eval_context["batch_id"] = (
                f"{self._logging_session}/rank-{self.args.process_index}/validation/"
                f"round-{self._eval_round}/{self._eval_context['dataset_key']}/"
                f"batch-{self._eval_batch_index}"
            )
        if has_alignment and self._eval_alignment_statistics is not None and "gap_distance" in outputs:
            self._eval_alignment_statistics.update(batch, outputs)

        self._record_eval_samples(
            forward_type,
            inputs[forward_type],
            outputs,
        )

        # Keep Trainer's scalar return contract; evaluation_loop corrects its
        # aggregate using the observed batch sizes above.
        return (outputs["loss"].detach(), None, None)

    def _record_eval_samples(self, forward_type, batch, outputs):
        """Buffer per-sample inputs, targets and losses for this batch."""
        if self.eval_sample_log_limit == 0:
            return

        per_sample_loss = outputs["per_sample_loss"].float().cpu().tolist()

        if forward_type == "alignment":
            language_keys = batch["lang_pair"]
            positive_cosine = (
                outputs["positive_cosine"].float().cpu().tolist()
            )
            detailed = {}
            if "gap_distance" in outputs:
                pending = {}
                selected = []
                for index, pair in enumerate(language_keys):
                    group = f"alignment/{pair}"
                    count = pending.get(group, len(self.eval_sample_buffer.get(group, [])))
                    if count < self.eval_sample_log_limit:
                        selected.append(index)
                        pending[group] = count + 1
                context = self._eval_context or {
                    **self._observation_context("validation"),
                    "batch_id": f"{self._logging_session}/validation/manual-{uuid4().hex}",
                }
                detailed = {row["batch_position"]: row for row in alignment_records(batch, outputs, selected, context)}
                if selected:
                    self._eval_batch_buffer.append(batch_record(batch, outputs, context))
        else:
            language_keys = batch["lang"]
            num_target_tokens = (
                outputs["per_sample_num_tokens"].cpu().tolist()
            )

        for index, language_key in enumerate(language_keys):
            records = self.eval_sample_buffer.setdefault(
                f"{forward_type}/{language_key}",
                [],
            )

            if len(records) >= self.eval_sample_log_limit:
                continue

            # The tensor index restarts in every batch; storage IDs must not.
            sample_index = len(records)
            if forward_type == "alignment":
                source_last_layer_embedding_key = f"{forward_type}/{language_key}/source_last_layer_embeddings_{self.state.global_step}_{sample_index}"
                target_last_layer_embedding_key = f"{forward_type}/{language_key}/target_last_layer_embeddings_{self.state.global_step}_{sample_index}"
                source_embedding_key = f"{forward_type}/{language_key}/source_embeddings_{self.state.global_step}_{sample_index}"
                target_embedding_key = f"{forward_type}/{language_key}/target_embeddings_{self.state.global_step}_{sample_index}"
                self.eval_sample_embedding_buffer[source_last_layer_embedding_key] = outputs["source_last_layer_embeddings"][index].detach().float().cpu().numpy().copy()
                self.eval_sample_embedding_buffer[target_last_layer_embedding_key] = outputs["target_last_layer_embeddings"][index].detach().float().cpu().numpy().copy()
                self.eval_sample_embedding_buffer[source_embedding_key] = outputs["source_embeddings"][index].detach().float().cpu().numpy().copy()
                self.eval_sample_embedding_buffer[target_embedding_key] = outputs["target_embeddings"][index].detach().float().cpu().numpy().copy()
                
                record = {
                    **detailed.get(index, {}),
                    "origin_data": batch["item"][index],
                    "source_text": batch["source_text"][index],
                    "target_text": batch["target_text"][index],
                    "loss": per_sample_loss[index],
                    "positive_cosine": positive_cosine[index],
                    "embedding_keys": {
                        "source_last_layer_embedding_key": source_last_layer_embedding_key,
                        "target_last_layer_embedding_key": target_last_layer_embedding_key,
                        "source_embedding_key": source_embedding_key,
                        "target_embedding_key": target_embedding_key,
                    }
                }
            else:
                utt_last_layer_embedding_key = f"{forward_type}/{language_key}/utt_last_layer_embeddings_{self.state.global_step}_{sample_index}"
                utt_embedding_key = f"{forward_type}/{language_key}/utt_embeddings_{self.state.global_step}_{sample_index}"
                self.eval_sample_embedding_buffer[utt_last_layer_embedding_key] = outputs["utt_last_layer_embeddings"][index].detach().float().cpu().numpy().copy()
                self.eval_sample_embedding_buffer[utt_embedding_key] = outputs["utt_embeddings"][index].detach().float().cpu().numpy().copy()
                
                record = {
                    **(self._eval_context or {}),
                    "sample_id": batch["sample_id"][index] if "sample_id" in batch else None,
                    "record_id": f"{self._eval_context['batch_id']}/sample-{index}" if self._eval_context else None,
                    "batch_position": index,
                    "actual_batch_size": len(language_keys),
                    "origin_data": batch["item"][index],
                    "utt": batch["utt"][index],
                    "target": batch["target"][index],
                    "embedding_inputs": {"utt": "prompt_with_gold_answer"},
                    "loss": per_sample_loss[index],
                    "num_target_tokens": num_target_tokens[index],
                    "embedding_keys": {
                        "utt_last_layer_embedding_key": utt_last_layer_embedding_key,
                        "utt_embedding_key": utt_embedding_key,
                    }
                }

            record["global_step"] = self.state.global_step
            records.append(record)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        self._evaluation_depth += 1
        try:
            metrics = super().evaluate(
                eval_dataset,
                ignore_keys,
                metric_key_prefix,
            )
        except Exception:
            if self._evaluation_depth == 1:
                self.eval_sample_buffer = {}
                self.eval_sample_embedding_buffer = {}
                self._eval_batch_buffer = []
                self._eval_metric_buffer = {}
            raise
        finally:
            self._evaluation_depth -= 1

        # Dict evaluation recurses with a dataset-specific prefix. Flush at
        # the outermost call, including user prefixes such as "probe".
        if self._evaluation_depth == 0:
            self.flush_eval_samples()

        return metrics

    def flush_eval_samples(self):
        """Write the buffered per-sample records and reset the buffer."""
        if self.args.process_index == 0:
            directory = Path(self.args.output_dir) / "eval_samples"
            if self._eval_metric_buffer:
                write_json(directory / f"step-{self.state.global_step}_metrics.json", self._eval_metric_buffer)
            if self._eval_batch_buffer:
                write_json(directory / f"step-{self.state.global_step}_batches.json", self._eval_batch_buffer)
        self._eval_metric_buffer = {}
        self._eval_batch_buffer = []
        if not self.eval_sample_buffer:
            return

        if self.args.process_index == 0:
            output_path = (
                Path(self.args.output_dir)
                / "eval_samples"
                / f"step-{self.state.global_step}.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as sample_file:
                json.dump(
                    self.eval_sample_buffer,
                    sample_file,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                
            #save embeddings
            embedding_output_path = (
                Path(self.args.output_dir)
                / "eval_samples"
                / f"step-{self.state.global_step}_embeddings.pkl"
            )
            with embedding_output_path.open("wb") as embedding_file:
                pickle.dump(self.eval_sample_embedding_buffer, embedding_file)

            logger.info(f"Saved validation samples to {output_path}")
            logger.info(f"Saved validation embeddings to {embedding_output_path}")

        self.eval_sample_buffer = {}
        self.eval_sample_embedding_buffer = {}
