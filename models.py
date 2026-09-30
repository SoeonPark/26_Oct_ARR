import torch
from torch import nn
from torch.nn import functional as F

from config import resolve_alignment_loss


def _distance_precision(embeddings):
    # Upcast before subtraction and norm. Keep float64 for numerical checks.
    if embeddings.dtype in (torch.float16, torch.bfloat16):
        return embeddings.float()
    return embeddings


@torch.no_grad()
def alignment_sample_metrics(source_embeddings, target_embeddings, gap_distances=None):
    """Detached diagnostics on raw embeddings, shared by all alignment losses."""
    source = _distance_precision(source_embeddings)
    target = _distance_precision(target_embeddings)
    distances = (
        (target - source).norm(p=2, dim=-1)
        if gap_distances is None
        else gap_distances.detach()
    )
    positive_cosine = (
        F.normalize(source, p=2, dim=-1) * F.normalize(target, p=2, dim=-1)
    ).sum(dim=-1)
    return {
        "gap_distance": distances.detach(),
        "gap_distance_mean": distances.mean().detach(),
        "source_norm": source.norm(p=2, dim=-1).detach(),
        "target_norm": target.norm(p=2, dim=-1).detach(),
        "positive_cosine": positive_cosine.detach(),
    }


class CustomModel(nn.Module):

    def __init__(
        self,
        config,
        basemodel,
    ):
        super().__init__()

        # `Trainer` and its integrations expect model.config to be a
        # PretrainedConfig with a to_dict() method. Keep experiment arguments
        # separately and attach a serializable snapshot to the base config.
        self.experiment_config = config
        self.config = basemodel.config
        self.config.experiment_config = vars(config).copy()
        self.basemodel = basemodel
        
        # Fot PEFT, let the `Trainer` knows the quantized model's state
        self.is_loaded_in_4bit = getattr(basemodel, "is_loaded_in_4bit", False)
        self.is_loaded_in_8bit = getattr(basemodel, "is_loaded_in_8bit", False)
        self.quantization_method = getattr(basemodel, "quantization_method", None)
        hf_device_map = getattr(basemodel, "hf_device_map", None)
        if hf_device_map is not None:
            self.hf_device_map = hf_device_map
        self.peft_config = getattr(basemodel, "peft_config", None)

    def num_parameters(self, only_trainable=False):
        if hasattr(self.basemodel, "get_nb_trainable_parameters"):
            trainable_parameters, total_parameters = (
                self.basemodel.get_nb_trainable_parameters()
            )
            return (
                trainable_parameters
                if only_trainable
                else total_parameters
            )

        parameters = self.parameters()
        if only_trainable:
            parameters = (
                parameter
                for parameter in parameters
                if parameter.requires_grad
            )
        return sum(parameter.numel() for parameter in parameters)

    def mean_pool(
        self,
        hidden_states,
        attention_mask,
    ):
        mask = attention_mask.unsqueeze(-1).to(
            dtype=hidden_states.dtype
        )

        summed = (
            hidden_states * mask
        ).sum(dim=1)

        denominator = (
            mask.sum(dim=1)
            .clamp_min(1e-9)
        )

        return summed / denominator

    def get_alignment_embeddings(
        self,
        hidden_states,
        attention_mask,
    ):

        if self.experiment_config.alignment_hidden_state_position == "last_token":

            last_position = (
                attention_mask.sum(dim=1) - 1
            )

            batch_idx = torch.arange(
                hidden_states.size(0),
                device=hidden_states.device,
            )

            embeddings = hidden_states[
                batch_idx,
                last_position,
            ]

        elif self.experiment_config.alignment_hidden_state_position == "mean":

            embeddings = self.mean_pool(
                hidden_states,
                attention_mask,
            )

        else:
            raise ValueError(
                "Invalid "
                "alignment_hidden_state_position: "
                f"{self.experiment_config.alignment_hidden_state_position}"
            )

        return embeddings

    def compute_alignment_loss(
        self,
        source_embeddings,
        target_embeddings,
        return_per_sample=False,
    ):
        raw_source_embeddings = source_embeddings
        raw_target_embeddings = target_embeddings
        source_embeddings = F.normalize(
            source_embeddings,
            p=2,
            dim=-1,
        )

        target_embeddings = F.normalize(
            target_embeddings,
            p=2,
            dim=-1,
        )

        temperature = getattr(
            self.experiment_config,
            "alignment_temperature",
            0.05,
        )

        logits = (
            source_embeddings
            @ target_embeddings.T
        ) / temperature

        return self._infonce_from_logits(
            logits, raw_source_embeddings, raw_target_embeddings, return_per_sample,
        )

    def _infonce_from_logits(
        self, logits, raw_source_embeddings, raw_target_embeddings, return_per_sample=False,
    ):
        labels = torch.arange(
            logits.size(0),
            device=logits.device,
        )

        source_to_target_loss = F.cross_entropy(
            logits,
            labels,
        )

        target_to_source_loss = F.cross_entropy(
            logits.T,
            labels,
        )

        loss = (
            source_to_target_loss
            + target_to_source_loss
        ) / 2

        if not return_per_sample:
            return loss

        # Per-query loss under this batch's negatives. Only comparable across
        # batches when the negative pool is fixed, which is why validation
        # builds one dataset per language pair.
        per_sample_loss = (
            F.cross_entropy(logits, labels, reduction="none")
            + F.cross_entropy(logits.T, labels, reduction="none")
        ) / 2

        return loss, {
            "per_sample_loss": per_sample_loss.detach(),
            **alignment_sample_metrics(raw_source_embeddings, raw_target_embeddings),
        }

    def compute_contrastive_variant_loss(
        self, source_embeddings, target_embeddings, language_pairs, return_per_sample=False,
    ):
        """Single-pair, microbatch references; diagonal positives and all B candidates."""
        assert source_embeddings.ndim == 2 and source_embeddings.shape == target_embeddings.shape
        assert len(source_embeddings) >= 2, "Alignment needs at least two translation pairs."
        assert language_pairs is not None and len(language_pairs) == len(source_embeddings)
        assert len(set(language_pairs)) == 1, "Alignment requires a single language pair."
        config = self.experiment_config
        loss_type = resolve_alignment_loss(config)
        # Keep score construction and CE out of AMP; retain float64 for gradcheck.
        with torch.autocast(device_type=source_embeddings.device.type, enabled=False):
            source = _distance_precision(source_embeddings)
            target = _distance_precision(target_embeddings)
            if loss_type == "centered_infonce":
                source_centered = source - source.mean(dim=0, keepdim=True)
                target_centered = target - target.mean(dim=0, keepdim=True)
                scores = (
                    F.normalize(source_centered, dim=-1, eps=1e-8)
                    @ F.normalize(target_centered, dim=-1, eps=1e-8).T
                )
            else:
                gaps = target.unsqueeze(0) - source.unsqueeze(1)
                if loss_type == "gap_distance_infonce":
                    distances = gaps.norm(dim=-1)
                    mean_distance = distances.diagonal().mean()
                    scale = getattr(config, "alignment_gap_scale", 1.0)
                    scores = -((distances - mean_distance) / scale).square()
                elif loss_type == "gap_direction_infonce":
                    mean_gap = (target - source).mean(dim=0)
                    scores = F.cosine_similarity(gaps, mean_gap[None, None, :], dim=-1, eps=1e-8)
                else:
                    raise ValueError(f"Not a contrastive variant: {loss_type}")
            logits = scores / getattr(config, "alignment_temperature", 0.05)
            return self._infonce_from_logits(logits, source, target, return_per_sample)

    
    # Our Proposed Gap Consistency Loss
    def compute_gap_consistency_loss(
        self,
        source_embeddings,
        target_embeddings,
        language_pairs,
        return_per_sample=False,
    ):
        """Population variance of raw Euclidean translation-pair distances.

        Equal distances incur zero loss regardless of gap direction. The
        reference distance is this microbatch's mean, not a preserved or fixed
        language-pair distance.
        """
        assert source_embeddings.ndim == target_embeddings.ndim == 2, (
            "Expected source and target embeddings with shape [batch, hidden]."
        )
        assert source_embeddings.shape == target_embeddings.shape, (
            "Source and target embedding shapes must match."
        )
        assert source_embeddings.size(0) >= 2, (
            "Gap consistency requires at least two translation pairs per microbatch."
        )
        assert language_pairs is not None, "Language-pair labels are required."
        assert len(language_pairs) == source_embeddings.size(0), (
            "Expected one language-pair label per sample."
        )
        unique_language_pairs = set(language_pairs)
        assert len(unique_language_pairs) == 1, (
            f"Expected one language pair, got {unique_language_pairs}."
        )

        source = _distance_precision(source_embeddings)
        target = _distance_precision(target_embeddings)
        gap_distances = (target - source).norm(p=2, dim=-1)
        mean_distance = gap_distances.mean()
        per_sample_loss = (gap_distances - mean_distance).pow(2)
        loss = per_sample_loss.mean()
        if not return_per_sample:
            return loss
        return loss, {
            "per_sample_loss": per_sample_loss.detach(),
            **alignment_sample_metrics(source, target, gap_distances),
        }

    def forward(
        self,
        forward_type="alignment",
        return_per_sample=False,
        **inputs,
    ):

        if forward_type == "alignment" or forward_type == "inference":

            data = inputs["alignment"]

            source_out = self.basemodel(
                input_ids=data["source_input_ids"],
                attention_mask=data[
                    "source_attention_mask"
                ],
                output_hidden_states=True,
                return_dict=True,
            )

            target_out = self.basemodel(
                input_ids=data["target_input_ids"],
                attention_mask=data[
                    "target_attention_mask"
                ],
                output_hidden_states=True,
                return_dict=True,
            )

            layer = (
                self.experiment_config.alignment_hidden_state_layer
            )

            source_hidden_states = (
                source_out.hidden_states[layer]
            )

            target_hidden_states = (
                target_out.hidden_states[layer]
            )

            source_embeddings = (
                self.get_alignment_embeddings(
                    source_hidden_states,
                    data["source_attention_mask"],
                )
            )

            target_embeddings = (
                self.get_alignment_embeddings(
                    target_hidden_states,
                    data["target_attention_mask"],
                )
            )
            
            source_last_layer_embeddings = (
                self.get_alignment_embeddings(
                    source_out.hidden_states[-1],
                    data["source_attention_mask"],
                )
            )
            
            target_last_layer_embeddings = (
                self.get_alignment_embeddings(
                    target_out.hidden_states[-1],
                    data["target_attention_mask"],
                )
            )

            alignment_loss_type = resolve_alignment_loss(self.experiment_config)
            if alignment_loss_type == "gap_consistency":
                loss_result = self.compute_gap_consistency_loss(
                    source_embeddings,
                    target_embeddings,
                    data.get("lang_pair"),
                    return_per_sample=return_per_sample,
                )
            elif alignment_loss_type == "infonce":
                loss_result = self.compute_alignment_loss(
                    source_embeddings,
                    target_embeddings,
                    return_per_sample=return_per_sample,
                )
            else:
                loss_result = self.compute_contrastive_variant_loss(
                    source_embeddings,
                    target_embeddings,
                    data.get("lang_pair"),
                    return_per_sample=return_per_sample,
                )
            if return_per_sample:
                alignment_loss, per_sample_values = loss_result
            else:
                alignment_loss, per_sample_values = loss_result, {}

            alignment_output = {
                "loss": alignment_loss,
                "alignment_loss_type": alignment_loss_type,
                "source_embeddings": source_embeddings,
                "target_embeddings": target_embeddings,
                "source_last_layer_embeddings": source_last_layer_embeddings,
                "target_last_layer_embeddings": target_last_layer_embeddings,
                "lang_pair": data.get("lang_pair"),
                **per_sample_values,
            }

        if forward_type == "downstream" or forward_type == "inference":

            data = inputs["downstream"]
            layer = self.experiment_config.alignment_hidden_state_layer

            output = self.basemodel(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
                labels=data["labels"],
                output_hidden_states=True,
                return_dict=True,
            )

            utt_embeddings = self.get_alignment_embeddings(
                output.hidden_states[layer],
                data["attention_mask"]
            )
            
            utt_last_layer_embeddings = self.get_alignment_embeddings(
                output.hidden_states[-1],
                data["attention_mask"]
            )
            
            downstream_output = {
                "loss": output.loss,
                "lang": data.get("lang"),
                "utt": data.get("utt"),
                "utt_embeddings": utt_embeddings,
                "utt_last_layer_embeddings": utt_last_layer_embeddings,
                "target": data.get("target"),
            }

            if return_per_sample:
                # Returning [B, T, vocab] logits to the Trainer would accumulate
                # hundreds of MB per eval batch, so reduce to per-sample values
                # here and drop the logits entirely.
                shift_logits = output.logits[:, :-1, :]
                shift_labels = data["labels"][:, 1:]
                valid = shift_labels != -100

                # Upcasting the whole [B, T, vocab] tensor to fp32 at once costs
                # about a gigabyte, so reduce one row at a time.
                row_losses = []
                for row in range(shift_logits.size(0)):
                    token_losses = F.cross_entropy(
                        shift_logits[row].float(),
                        shift_labels[row],
                        reduction="none",
                        ignore_index=-100,
                    )
                    row_losses.append(
                        token_losses.sum()
                        / valid[row].sum().clamp_min(1)
                    )

                downstream_output["per_sample_loss"] = (
                    torch.stack(row_losses).detach()
                )
                downstream_output["per_sample_num_tokens"] = (
                    valid.sum(dim=1).detach()
                )
            else:
                downstream_output["logits"] = output.logits

        if forward_type == "alignment":
            return alignment_output
        elif forward_type == "downstream":
            return downstream_output
        elif forward_type == "inference":
            return {
                "alignment": alignment_output,
                "downstream": downstream_output,
            }
