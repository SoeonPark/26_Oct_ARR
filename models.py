import torch
from torch import nn
from torch.nn import functional as F


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

        # source/target embeddings were reassigned to their normalized form
        # above, so this dot product is the positive pair's cosine similarity.
        positive_cosine = (
            source_embeddings * target_embeddings
        ).sum(dim=-1)

        return loss, {
            "per_sample_loss": per_sample_loss.detach(),
            "positive_cosine": positive_cosine.detach(),
        }

    
    # Our Proposed Gap Consistency Loss
    def compute_gap_consistency_loss(
        self,
        source_embeddings,
        target_embeddings,
        language_pairs,
    ):
        gap_vectors = target_embeddings - source_embeddings

        unique_language_pairs = []
        for language_pair in language_pairs:
            if language_pair not in unique_language_pairs:
                unique_language_pairs.append(language_pair)

        pair_losses = []

        for language_pair in unique_language_pairs:
            pair_mask = torch.tensor(
                [
                    current_pair == language_pair
                    for current_pair in language_pairs
                ],
                device=gap_vectors.device,
                dtype=torch.bool,
            )
            pair_gap_vectors = gap_vectors[pair_mask]
            mean_pair_gap = pair_gap_vectors.mean(
                dim=0,
                keepdim=True,
            )
            centered_pair_gaps = pair_gap_vectors - mean_pair_gap

            # Mean squared L2 distance from the language-pair mean offset.
            pair_loss = (
                centered_pair_gaps.pow(2)
                .sum(dim=-1)
                .mean()
            )
            pair_losses.append(pair_loss)

        return torch.stack(pair_losses).mean()

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

            if return_per_sample:
                info_nce_loss, per_sample_values = (
                    self.compute_alignment_loss(
                        source_embeddings,
                        target_embeddings,
                        return_per_sample=True,
                    )
                )
            else:
                info_nce_loss = (
                    self.compute_alignment_loss(
                        source_embeddings,
                        target_embeddings,
                    )
                )
                per_sample_values = {}

            alignment_output = {
                "loss": info_nce_loss,
                "source_embeddings": source_embeddings,
                "target_embeddings": target_embeddings,
                "lang_pair": data.get("lang_pair"),
                **per_sample_values,
            }

        if forward_type == "downstream" or forward_type == "inference":

            data = inputs["downstream"]

            output = self.basemodel(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
                labels=data["labels"],
                return_dict=True,
            )

            downstream_output = {
                "loss": output.loss,
                "lang": data.get("lang"),
                "utt": data.get("utt"),
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
