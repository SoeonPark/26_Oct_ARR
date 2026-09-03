import json
import os
from pathlib import Path

import torch
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils import logging


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
        **kwargs,
    ):
        self.schedule = schedule
        self.eval_forward_type = eval_forward_type
        self.training_type = training_type
        self.total_steps = total_steps

        # Per-sample validation records, keyed by "<objective>/<language>".
        # Filled in prediction_step and written out once per evaluation round.
        self.eval_sample_log_limit = eval_sample_log_limit
        self.eval_sample_buffer = {}

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

    def objective_for_step(self):
        if self.training_type == "transfer_only":
            return "downstream"
        elif self.training_type == "contrastive_only":
            return "alignment"
        elif self.training_type == "contrastive_then_transfer":
            half_steps = self.total_steps // 2
            if self.state.global_step < half_steps:
                return "alignment"
            else:
                return "downstream"
        elif self.training_type == "alternative":
            return self.schedule[
                self.state.global_step
                % len(self.schedule)
            ]
        else:
            raise ValueError(
                f"Unknown training_type: {self.training_type}"
            )

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
        inputs["forward_type"] = objective

        loss = super().training_step(
            model,
            inputs,
            num_items_in_batch,
        )

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

        self._record_eval_samples(
            forward_type,
            inputs[forward_type],
            outputs,
        )

        # evaluation_loop weights this by find_batch_size(inputs), so keep it a
        # 0-dim tensor rather than a Python float.
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

            if forward_type == "alignment":
                record = {
                    "source_text": batch["source_text"][index],
                    "target_text": batch["target_text"][index],
                    "loss": per_sample_loss[index],
                    "positive_cosine": positive_cosine[index],
                }
            else:
                record = {
                    "utt": batch["utt"][index],
                    "target": batch["target"][index],
                    "loss": per_sample_loss[index],
                    "num_target_tokens": num_target_tokens[index],
                }

            record["global_step"] = self.state.global_step
            records.append(record)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        metrics = super().evaluate(
            eval_dataset,
            ignore_keys,
            metric_key_prefix,
        )

        # A dict eval_dataset makes Trainer.evaluate recurse once per dataset
        # with metric_key_prefix=f"eval_{name}". Flushing only on the outer
        # call therefore writes one file per evaluation round.
        if metric_key_prefix == "eval":
            self.flush_eval_samples()

        return metrics

    def flush_eval_samples(self):
        """Write the buffered per-sample records and reset the buffer."""
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

            logger.info(f"Saved validation samples to {output_path}")

        self.eval_sample_buffer = {}
