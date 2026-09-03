# utils.py
import os
import json

MASSIVE_LANG_MAP = {
    "en": "en-US",
    "af": "af-ZA",
    "am": "am-ET",
    "ar": "ar-SA",
    "az": "az-AZ",
    "bn": "bn-BD",
    "cy": "cy-GB",
    "da": "da-DK",
    "de": "de-DE",
    "el": "el-GR",
    "es": "es-ES",
    "fa": "fa-IR",
    "fi": "fi-FI",
    "fr": "fr-FR",
    "he": "he-IL",
    "hi": "hi-IN",
    "hu": "hu-HU",
    "hy": "hy-AM",
    "id": "id-ID",
    "is": "is-IS",
    "it": "it-IT",
    "ja": "ja-JP",
    "jv": "jv-ID",
    "ka": "ka-GE",
    "km": "km-KH",
    "kn": "kn-IN",
    "ko": "ko-KR",
    "lv": "lv-LV",
    "ml": "ml-IN",
    "mn": "mn-MN",
    "pl": "pl-PL",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "sw": "sw-KE",
    "th": "th-TH",
    "tl": "tl-PH",
    "tr": "tr-TR",
    "ur": "ur-PK",
    "zh": "zh-CN",
}

MASSIVE_SYSTEM_PROMPT = (
    "Given a command from the user, a voice assistant will extract entities "
    "essential for carry out the command. "
    "Your task is to extract the entities as words from the command if they "
    "fall under a predefined list of entity types."
)


# LoRA target modules per Hugging Face model_type.
#
# PEFT 0.20 maps llama/qwen2/qwen3 to ["q_proj", "v_proj"] and has no entry at
# all for qwen3_5, where get_peft_model raises rather than guess. These lists
# follow the QLoRA recommendation of adapting every linear layer in the
# transformer block instead, since adapter count matters more than rank for
# matching full finetuning.
#
# Keeping this explicit rather than auto-detecting nn.Linear also makes the
# choice reproducible: the resolved list is what goes into the paper appendix,
# and it does not shift when a transformers release renames a submodule.
_ATTENTION_AND_MLP = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LORA_TARGET_MODULES = {
    "llama": list(_ATTENTION_AND_MLP),
    "qwen2": list(_ATTENTION_AND_MLP),
    "qwen3": list(_ATTENTION_AND_MLP),

    # Qwen3.5 interleaves linear_attention and full_attention blocks
    # (full_attention_interval=4). Only the full_attention blocks carry
    # q/k/v/o_proj, so targeting those alone reaches a quarter of the layers.
    # The linear_attention blocks are covered through their own projections.
    #
    # in_proj_a and in_proj_b are excluded on purpose: they map hidden_size to
    # one value per head (2048 -> 16), so an r=16 adapter on them would not be
    # low rank at all.
    "qwen3_5": _ATTENTION_AND_MLP + [
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
    ],
}


def resolve_lora_target_modules(model_type):
    """Return the LoRA target module names for a Hugging Face model_type."""
    if model_type not in LORA_TARGET_MODULES:
        raise ValueError(
            f"No LoRA target modules registered for model_type "
            f"'{model_type}'. Known types: "
            f"{sorted(LORA_TARGET_MODULES)}. Add an entry to "
            f"LORA_TARGET_MODULES in utils.py, or pass "
            f"--peft_target_modules explicitly."
        )

    return list(LORA_TARGET_MODULES[model_type])
