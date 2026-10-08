"""COMET row correspondence, aggregation and isolated evaluation; no model download."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from comet22_scoring import (
    Comet22Scorer, checked_scores, comet_inputs, read_jsonl, score_comet22,
    score_files, summarize_scores, write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def example(sample_id, direction="en-de", prediction="Hallo!"):
    return {
        "sample_id": sample_id, "direction": direction, "lang": direction,
        "source": "Hello!", "prediction": prediction, "target": "Hallo!",
        "generation_limit_reached": False, "source_paragraphs": 1, "prediction_paragraphs": 1,
    }


class Comet22Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_text_order_and_empty_hypothesis_are_preserved(self):
        rows = [example("b", prediction=""), example("a", "de-en", "  hello\n世界 ")]
        rows[1].update(source="  Grüß\n世界 ", target="  Hello\n世界 ")
        self.assertEqual(comet_inputs(rows), [
            {"src": "Hello!", "mt": "", "ref": "Hallo!"},
            {"src": "  Grüß\n世界 ", "mt": "  hello\n世界 ", "ref": "  Hello\n世界 "},
        ])

    def test_invalid_inputs_and_scores_fail_instead_of_silently_dropping_rows(self):
        for key, value in (("source", ""), ("target", " \n"), ("prediction", None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                comet_inputs([{**example("1"), key: value}])
        with self.assertRaises(ValueError):
            comet_inputs([])
        for scores in ([0.5], [0.5, float("nan")], [float("inf"), 0.5]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                checked_scores(scores, 2)
        # Do not clip regression outputs to [0, 1].
        self.assertEqual(checked_scores([-0.01, 1.01], 2), [-0.01, 1.01])

    def test_macro_is_not_segment_mean_and_directions_are_separate(self):
        rows = [example("1"), example("2"), example("3", "en-ja"), example("4", "de-en")]
        report = summarize_scores(rows, [0.2, 0.4, 0.9, 0.6], {"fixture": True})
        self.assertAlmostEqual(report["mean_score"], 0.525)
        self.assertAlmostEqual(report["macro_average"]["comet22"], 0.6)
        self.assertAlmostEqual(report["by_language"]["en-de"]["comet22_x100"], 30)
        self.assertAlmostEqual(report["macro_by_translation_direction"]["en_to_x"]["comet22"], 0.6)
        self.assertEqual(report["macro_by_translation_direction"]["x_to_en"]["num_directions"], 1)
        legacy = example("1")
        legacy.pop("direction")
        legacy["lang"] = "de"
        self.assertIn("en-de", summarize_scores([legacy], [0.5], {})["by_language"])

    def test_model_receives_reference_triplets_and_zero_dropout(self):
        scorer = Comet22Scorer.__new__(Comet22Scorer)
        scorer.batch_size, scorer.gpus, scorer.metadata = 16, 0, {}
        calls = []

        def predict(samples, **kwargs):
            calls.append((samples, kwargs))
            return SimpleNamespace(scores=[0.8, 0.1], system_score=0.45)

        scorer.model = SimpleNamespace(predict=predict)
        rows = [example("1"), example("2", prediction="")]
        scores, metadata = scorer.score(rows)
        self.assertEqual(scores, [0.8, 0.1])
        self.assertEqual(calls[0][0], comet_inputs(rows))
        self.assertEqual(calls[0][1]["mc_dropout"], 0)
        self.assertEqual(calls[0][1]["accelerator"], "cpu")
        self.assertEqual(metadata["system_score"], 0.45)

    def test_saved_scores_keep_sample_correspondence_and_bleu_report(self):
        paths = [self.root / "wmt23_predictions.en-ja.jsonl", self.root / "wmt23_predictions.en-de.jsonl"]
        write_jsonl(paths[0], [example("j", "en-ja")])
        write_jsonl(paths[1], [example("d")])
        bleu = self.root / "wmt23_metrics.json"
        bleu.write_text('{"metric":"sacrebleu"}')
        with patch("comet22_scoring.Comet22Scorer") as scorer:
            scorer.return_value.score.return_value = ([0.7, 0.9], {"test": True})
            report = score_files(paths, self.root)
        saved = read_jsonl(self.root / "wmt23_comet22_scores.jsonl")
        self.assertEqual([(row["sample_id"], row["comet22"]) for row in saved], [("j", 0.7), ("d", 0.9)])
        self.assertEqual(report["prediction_files"][0]["num_examples"], 1)
        self.assertEqual(len(report["prediction_files"][0]["sha256"]), 64)
        self.assertEqual(bleu.read_text(), '{"metric":"sacrebleu"}')

    def test_duplicate_or_missing_samples_fail_before_model_load(self):
        path = self.root / "predictions.jsonl"
        for rows in ([example("duplicate"), example("duplicate")], [{**example("x"), "sample_id": None}]):
            write_jsonl(path, rows)
            with patch("comet22_scoring.Comet22Scorer") as scorer, self.assertRaises(ValueError):
                score_files([path], self.root)
            scorer.assert_not_called()

    def test_scoring_failure_does_not_publish_results(self):
        path = self.root / "predictions.jsonl"
        write_jsonl(path, [example("1")])
        with patch("comet22_scoring.Comet22Scorer") as scorer:
            scorer.return_value.score.side_effect = RuntimeError("out of memory")
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                score_files([path], self.root)
        self.assertFalse((self.root / "wmt23_comet22_metrics.json").exists())

    def test_subprocess_bridge_preserves_scores_and_propagates_failure(self):
        rows = [example("a"), example("b", prediction="")]
        args = SimpleNamespace(comet22_python="/separate/python", comet22_checkpoint="/official/model.ckpt",
                               comet22_batch_size=4, comet22_gpus=0)

        def run(command, check):
            self.assertTrue(check)
            self.assertEqual(command[0], "/separate/python")
            self.assertEqual(command[-2:], ["--checkpoint", "/official/model.ckpt"])
            source = Path(command[command.index("--predictions") + 1])
            self.assertEqual(read_jsonl(source), rows)
            write_jsonl(source.parent / "wmt23_comet22_scores.jsonl",
                        [{**row, "comet22": score} for row, score in zip(rows, [0.7, 0.2])])
            (source.parent / "wmt23_comet22_metrics.json").write_text(json.dumps({"scorer_metadata": {"isolated": True}}))

        with patch("comet22_scoring.subprocess.run", side_effect=run):
            self.assertEqual(score_comet22(rows, args), ([0.7, 0.2], {"isolated": True}))

        def reordered(command, check):
            run(command, check)
            root = Path(command[command.index("--output_dir") + 1])
            output = root / "wmt23_comet22_scores.jsonl"
            write_jsonl(output, reversed(read_jsonl(output)))

        with patch("comet22_scoring.subprocess.run", side_effect=reordered):
            with self.assertRaisesRegex(ValueError, "row order"):
                score_comet22(rows, args)
        with patch("comet22_scoring.subprocess.run", side_effect=subprocess.CalledProcessError(1, "python")):
            with self.assertRaises(subprocess.CalledProcessError):
                score_comet22(rows, args)

    def test_launcher_runs_comet_after_generation_and_for_both_scopes(self):
        for name in ("experiment_config.json", "adapter_config.json", "adapter_model.bin"):
            (self.root / name).touch()
        env = {**os.environ, "EVAL_COMET22": "true", "EVAL_TASKS": "wmt23", "EVAL_SPLIT": "test",
               "EVAL_LANGUAGE_SCOPE": "both", "EVAL_OUTPUT_DIR": str(self.root / "output with spaces"),
               "COMET22_PYTHON": sys.executable, "COMET22_GPUS": "0", "EVAL_WMT23_METRIC": "sacrebleu"}
        result = subprocess.run(["bash", str(ROOT / "scripts/wmt23_eval.sh"), "--dry-run", str(self.root)],
                                env=env, check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.count("score_comet22.py"), 2)
        self.assertLess(result.stdout.index("evaluate.py"), result.stdout.index("score_comet22.py"))
        self.assertIn("--wmt23_metric sacrebleu", result.stdout)
        self.assertIn("/test/in", result.stdout)
        self.assertIn("/test/out", result.stdout)
        env["EVAL_WMT23_METRIC"] = "comet22"
        result = subprocess.run(["bash", str(ROOT / "scripts/wmt23_eval.sh"), "--dry-run", str(self.root)],
                                env=env, check=True, capture_output=True, text=True)
        self.assertIn("--wmt23_metric none", result.stdout)


if __name__ == "__main__":
    unittest.main()
