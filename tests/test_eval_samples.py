"""Sample/vector identity checks without checkpoints, downloads, or GPUs."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM

from custom_trainer import AlternativeRoutingTrainer
from alignment_logging import AlignmentStatistics
from data_utils import make_sample_id
from evaluate import (
    EvalSampleRecorder,
    build_massive_evaluation_samples,
    collate_massive_evaluation_samples,
    collect_alignment_embeddings,
    collect_massive_sample_embeddings,
    evaluate_massive_predictions,
    generate_massive_predictions,
    load_experiment_config,
    parse_eval_args,
)
from models import CustomModel


class NumericTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    def __call__(self, texts, add_special_tokens=True, **kwargs):
        rows = [[int(word) for word in text.split()] for text in texts]
        if add_special_tokens:
            rows = [[99, *row] for row in rows]
        width = max(map(len, rows))
        ids, masks = [], []
        for row in rows:
            padding = [0] * (width - len(row))
            if self.padding_side == "left":
                ids.append(padding + row)
                masks.append(padding + [1] * len(row))
            else:
                ids.append(row + padding)
                masks.append([1] * len(row) + padding)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    def batch_decode(self, ids, **kwargs):
        return ["slot: predicted"] * len(ids)


class MarkerBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.embedding = torch.nn.Embedding(256, 3)
        self.forward_inputs = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, **kwargs):
        self.forward_inputs.append((input_ids.clone(), attention_mask.clone()))
        positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        hidden = torch.stack((input_ids, positions, attention_mask), dim=-1).float()
        return SimpleNamespace(hidden_states=(hidden, hidden + 10, hidden + 20))

    def generate(self, input_ids, **kwargs):
        return torch.cat((input_ids, torch.full((len(input_ids), 1), 7)), dim=1)


def make_model(base=None, pooling="last_token"):
    return CustomModel(
        SimpleNamespace(
            alignment_hidden_state_layer=1,
            alignment_hidden_state_position=pooling,
            alignment_temperature=0.05,
        ),
        base if base is not None else MarkerBase(),
    ).eval()


class NumericMassive:
    def __init__(self):
        self.config = SimpleNamespace(downstream_task_data="numeric-massive")
        self.lang_map = {"en": "en-US", "ko": "ko-KR"}
        self.split = "in_validation"
        self.SPLIT_MAPPING = {"in_validation": "validation"}
        self.all_data = {
            language: [
                {"id": index, "intent": index % 3,
                 "utt": f"10 {index + 20}" if index % 2 else str(index + 20),
                 "annot_utt": "slot: gold"}
                for index in range(8)
            ]
            for language in ("en", "ko")
        }

    @staticmethod
    def extract_slots(annotation):
        return annotation

    @staticmethod
    def _apply_chat_template(utterance, target=None):
        if target is not None:
            raise AssertionError("Gold answers must not enter representation inputs")
        return f"100 {utterance} 200", None


class EvalSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_trainer_64_samples_keep_64_vectors_per_type_and_flush_both_buffers(self):
        for objective, language, names in (
            ("alignment", "en-ko", ("source_embeddings", "target_embeddings",
             "source_last_layer_embeddings", "target_last_layer_embeddings")),
            ("downstream", "en", ("utt_embeddings", "utt_last_layer_embeddings")),
        ):
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as tmp:
                trainer = AlternativeRoutingTrainer.__new__(AlternativeRoutingTrainer)
                trainer.eval_sample_log_limit = 64
                trainer.eval_sample_buffer = {}
                trainer.eval_sample_embedding_buffer = {}
                trainer._eval_metric_buffer = {}
                trainer._eval_batch_buffer = []
                trainer._eval_context = None
                trainer.args = SimpleNamespace(output_dir=tmp, process_index=0)
                trainer.state = SimpleNamespace(global_step=500)
                for step in (500, 1000):
                    trainer.state.global_step = step
                    for start in range(0, 80, 16):
                        trainer._eval_context = {
                            "batch_id": f"fixture/{step}/{start}",
                            "global_step": step,
                            "phase": "validation",
                        }
                        markers = torch.arange(start, start + 16, dtype=torch.bfloat16)
                        texts = [str(int(marker)) for marker in markers]
                        batch = {"item": [{"id": text} for text in texts],
                                 "lang_pair": [language] * 16, "lang": [language] * 16,
                                 "source_text": texts, "target_text": texts,
                                 "utt": texts, "target": texts}
                        batch["sample_id"] = [f"fixture/{language}/{text}" for text in texts]
                        for side in ("source", "target"):
                            batch[f"{side}_input_ids"] = markers.long()[:, None]
                            batch[f"{side}_attention_mask"] = torch.ones(16, 1, dtype=torch.long)
                        outputs = {name: markers[:, None] for name in names}
                        outputs.update(per_sample_loss=markers, positive_cosine=markers,
                                       per_sample_num_tokens=markers)
                        outputs.update(
                            alignment_loss_type="infonce", loss=markers.float().mean(),
                            gap_distance=markers.float(), source_norm=markers.float(),
                            target_norm=markers.float(), gap_distance_mean=markers.float().mean(),
                        )
                        trainer._record_eval_samples(objective, batch, outputs)
                    trainer.flush_eval_samples()
                    self.assertEqual(trainer.eval_sample_buffer, {})
                    self.assertEqual(trainer.eval_sample_embedding_buffer, {})
                    records = json.loads((Path(tmp) / f"eval_samples/step-{step}.json").read_text())
                    with (Path(tmp) / f"eval_samples/step-{step}_embeddings.pkl").open("rb") as file:
                        vectors = pickle.load(file)
                    records = records[f"{objective}/{language}"]
                    self.assertEqual(len(records), 64)
                    self.assertEqual(len(vectors), 64 * len(names))
                    for index, record in enumerate(records):
                        self.assertEqual(record["origin_data"]["id"], str(index))
                        for key in record["embedding_keys"].values():
                            self.assertEqual(vectors[key].tolist(), [float(index)])
                            self.assertIn(f"_{step}_", key)

    def test_alignment_exports_both_layers_across_batches_without_limiting_retrieval(self):
        model = make_model()
        batches = []
        for start in (0, 16, 32, 48, 64):
            ids = torch.arange(start + 1, start + 17)[:, None]
            batches.append({
                "source_input_ids": ids, "target_input_ids": ids + 100,
                "source_attention_mask": torch.ones_like(ids),
                "target_attention_mask": torch.ones_like(ids),
                "lang_pair": ["en-ko"] * 16,
                "source_text": [str(i) for i in range(start + 1, start + 17)],
                "target_text": [str(i) for i in range(start + 101, start + 117)],
                "item": [{"id": i} for i in range(start + 1, start + 17)],
                "sample_id": [f"numeric-opus/en-ko/validation/{i}" for i in range(start + 1, start + 17)],
            })
        recorder = EvalSampleRecorder(64)
        with redirect_stdout(io.StringIO()):
            groups = collect_alignment_embeddings(model, batches, recorder)
        self.assertEqual(len(groups["en-ko"]["source"]), 80)
        self.assertEqual(len(recorder.embeddings), 64 * 4)
        for index, record in enumerate(recorder.records["alignment/en-ko"], start=1):
            self.assertEqual(record["origin_data"]["id"], index)
            for name, offset in (("source_embedding_key", 10),
                                 ("target_embedding_key", 110),
                                 ("source_last_layer_embedding_key", 20),
                                 ("target_last_layer_embedding_key", 120)):
                self.assertEqual(recorder.embeddings[record["embedding_keys"][name]][0], index + offset)

    def test_massive_sample_limit_mixed_batches_predictions_and_padding(self):
        model, tokenizer, dataset = make_model(), NumericTokenizer(), NumericMassive()
        loader = DataLoader(build_massive_evaluation_samples(dataset), batch_size=3,
                            collate_fn=collate_massive_evaluation_samples)
        recorder = EvalSampleRecorder(5)
        with redirect_stdout(io.StringIO()):
            recorded = generate_massive_predictions(
                model, tokenizer, dataset, loader, 1, recorder,
                context={"session_id": "fixture", "split": "validation", "scope": "in", "global_step": 500},
            )
            baseline = generate_massive_predictions(model, tokenizer, dataset, loader, 1)
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(len(recorded), 16)
        self.assertEqual(
            recorded[0]["sample_id"],
            make_sample_id("numeric-massive", "en-US", "validation", 0),
        )
        self.assertEqual([r["prediction"] for r in recorded], [r["prediction"] for r in baseline])
        self.assertEqual([r["sample_id"] for r in recorded], [r["sample_id"] for r in baseline])
        self.assertEqual(len({r["record_id"] for r in recorded}), len(recorded))
        self.assertNotEqual(recorded[0]["record_id"], baseline[0]["record_id"])
        self.assertEqual(recorded[0]["session_id"], "fixture")
        self.assertEqual(recorded[0]["global_step"], 500)
        self.assertEqual(recorded[0]["actual_batch_size"], 3)
        self.assertEqual(evaluate_massive_predictions(recorded), evaluate_massive_predictions(baseline))
        self.assertEqual(len(recorder.embeddings), 2 * 5 * 4)
        # Five batches need extraction, two forward passes each. Later samples
        # in a language and recording-disabled generation add no forward calls.
        self.assertEqual(len(model.basemodel.forward_inputs), 10)
        for language in ("en", "ko"):
            records = recorder.records[f"downstream/{language}"]
            self.assertEqual([r["origin_data"]["id"] for r in records], list(range(5)))
            for record in records:
                keys = record["embedding_keys"]
                last_word = int(record["utt"].split()[-1])
                self.assertEqual(recorder.embeddings[keys["utt_embedding_key"]][0], last_word + 10)
                self.assertEqual(recorder.embeddings[keys["utt_last_layer_embedding_key"]][0], last_word + 20)
                prompt = recorder.embeddings[keys["prompt_embedding_key"]]
                self.assertEqual(prompt[0], 210)
                self.assertEqual(prompt[1], len(record["prompt_text"].split()) - 1 + 10)
                prediction = next(p for p in recorded if p.get("sample_id") == record["sample_id"])
                self.assertEqual(prediction["embedding_keys"], keys)
                self.assertEqual(prediction["record_id"], record["record_id"])

    def test_alignment_full_diagnostics_are_independent_of_sample_limit(self):
        model = make_model()
        model.experiment_config.alignment_loss = "gap_consistency"
        batches = []
        for batch_index, distances in enumerate(([1, 3], [4, 8, 9])):
            source = torch.arange(10, 10 + len(distances))[:, None]
            target = source + torch.tensor(distances)[:, None]
            batches.append({
                "source_input_ids": source, "target_input_ids": target,
                "source_attention_mask": torch.ones_like(source),
                "target_attention_mask": torch.ones_like(target),
                "lang_pair": ["en-ko"] * len(distances),
                "sample_id": [f"pair/{batch_index}/{i}" for i in range(len(distances))],
                "source_text": [str(value.item()) for value in source],
                "target_text": [str(value.item()) for value in target],
                "item": [{"id": f"{batch_index}/{i}"} for i in range(len(distances))],
            })
        context = {"session_id": "fixture", "run_id": "run", "split": "test", "scope": "in"}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp) / "alignment_samples.jsonl"
            diagnostics = AlignmentStatistics()
            recorder = EvalSampleRecorder(1)
            recorded = collect_alignment_embeddings(
                model, batches, recorder, diagnostics, path, context,
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]
            batch_records = [
                json.loads(line)
                for line in path.with_name("alignment_batches.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 5)
            self.assertEqual(len(recorder.records["alignment/en-ko"]), 1)
            self.assertEqual([r["actual_batch_size"] for r in batch_records], [2, 3])
            self.assertEqual(batch_records[1]["sample_ids"], batches[1]["sample_id"])
            self.assertEqual(len({record["record_id"] for record in records}), 5)
            for record in records:
                self.assertAlmostEqual(record["per_sample_loss"], record["gap_squared_residual"])
                self.assertEqual(record["source_num_tokens"], 1)
            summary = diagnostics.summary()["en-ko"]
            self.assertEqual(summary["num_examples"], 5)
            self.assertAlmostEqual(summary["corpus_gap_distance_variance"], 9.2)
            self.assertAlmostEqual(summary["batch_gap_loss_mean"], 3.2)

            disabled_stats = AlignmentStatistics()
            baseline = collect_alignment_embeddings(
                model, batches, EvalSampleRecorder(0), disabled_stats,
            )
            self.assertEqual(diagnostics.summary(), disabled_stats.summary())
            for side in ("source", "target"):
                torch.testing.assert_close(recorded["en-ko"][side], baseline["en-ko"][side])
            bounded_only = EvalSampleRecorder(1)
            collect_alignment_embeddings(model, batches, bounded_only, context=context)
            bounded_record = bounded_only.records["alignment/en-ko"][0]
            self.assertEqual(bounded_record["batch_sample_ids"], batches[0]["sample_id"])
            self.assertEqual(bounded_record["batch_language_pairs"], batches[0]["lang_pair"])
            self.assertEqual(bounded_record["loss"], bounded_record["per_sample_loss"])
            # Re-running into one output path replaces scalar logs rather than
            # duplicating every observation from the prior evaluation.
            collect_alignment_embeddings(model, batches, sample_metrics_path=path, context=context)
            self.assertEqual(len(path.read_text().splitlines()), 5)

    def test_saved_loss_defaults_and_scalar_logging_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment_config.json"
            path.write_text("{}")
            self.assertEqual(load_experiment_config(Path(tmp)).alignment_loss, "infonce")
            path.write_text('{"alignment_loss": "gap_consistency"}')
            self.assertEqual(load_experiment_config(Path(tmp)).alignment_loss, "gap_consistency")
            path.write_text('{"alignment_loss": "invalid"}')
            with self.assertRaises(ValueError):
                load_experiment_config(Path(tmp))
        with patch("sys.argv", ["evaluate.py", "--checkpoint_path", "unused"]):
            self.assertTrue(parse_eval_args().save_alignment_sample_metrics)
        with patch("sys.argv", ["evaluate.py", "--checkpoint_path", "unused", "--no-save_alignment_sample_metrics"]):
            self.assertFalse(parse_eval_args().save_alignment_sample_metrics)

    def test_zero_limit_skips_extra_forwards_and_sample_files(self):
        model, tokenizer, dataset = make_model(), NumericTokenizer(), NumericMassive()
        loader = DataLoader(build_massive_evaluation_samples(dataset), batch_size=4,
                            collate_fn=collate_massive_evaluation_samples)
        recorder = EvalSampleRecorder(0)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            predictions = generate_massive_predictions(model, tokenizer, dataset, loader, 1, recorder)
            recorder.save(Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertEqual(len(predictions), 16)
        self.assertEqual(model.basemodel.forward_inputs, [])

    def test_inference_json_pickle_round_trip_and_scope_isolation(self):
        recorder = EvalSampleRecorder(2)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            for language in ("en", "fr"):
                for index in range(2):
                    recorder.add("downstream", language, {"utt": str(index)},
                                 {"utt_embeddings": torch.tensor([index], dtype=torch.bfloat16)})
                output_dir = Path(tmp) / language
                recorder.save(output_dir)
                self.assertEqual(recorder.records, {})
                self.assertEqual(recorder.embeddings, {})
                records = json.loads((output_dir / "eval_samples.json").read_text())
                with (output_dir / "eval_samples_embeddings.pkl").open("rb") as file:
                    vectors = pickle.load(file)
                self.assertEqual(list(records), [f"downstream/{language}"])
                self.assertEqual(len(vectors), 2)
                for index, record in enumerate(records[f"downstream/{language}"]):
                    self.assertEqual(vectors[record["embedding_keys"]["utt_embedding_key"]].tolist(), [index])

    def test_prompt_embeddings_match_real_generation_prefill_with_left_padding(self):
        torch.manual_seed(17)
        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=256, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        ))
        model = make_model(base)
        tokenizer = NumericTokenizer()
        tokenizer.padding_side = "left"
        utterances = ["20", "10 21 22"]
        prompts = tokenizer([f"100 {utt} 200" for utt in utterances], add_special_tokens=False)
        with torch.inference_mode():
            generated = base.generate(**prompts, max_new_tokens=1, do_sample=False,
                                      output_hidden_states=True, return_dict_in_generate=True)
            for pooling in ("last_token", "mean"):
                with self.subTest(pooling=pooling):
                    model.experiment_config.alignment_hidden_state_position = pooling
                    embeddings = collect_massive_sample_embeddings(model, tokenizer, utterances, prompts, [0, 1])
                    self.assertEqual(tokenizer.padding_side, "left")
                    for name, layer in (("prompt_embeddings", 1), ("prompt_last_layer_embeddings", -1)):
                        hidden = generated.hidden_states[0][layer]
                        if pooling == "last_token":
                            expected = hidden[:, -1, :]
                        else:
                            mask = prompts["attention_mask"].unsqueeze(-1)
                            expected = (hidden * mask).sum(1) / mask.sum(1)
                        torch.testing.assert_close(embeddings[name], expected, rtol=1e-5, atol=1e-6)
            after_extraction = base.generate(**prompts, max_new_tokens=1, do_sample=False)
            torch.testing.assert_close(after_extraction, generated.sequences)


if __name__ == "__main__":
    unittest.main()
