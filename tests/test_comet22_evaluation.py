"""COMET integration with MT generation and saved evaluation metrics."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluate as mt_evaluate


class CometEvaluationTests(unittest.TestCase):
    def test_generation_scope_accepts_comet_and_saves_per_direction_scores(self):
        rows = [
            {"sample_id": "1", "lang": "en-de", "source": "Hello", "target": "Hallo", "prediction": "Hallo",
             "generation_limit_reached": False, "source_paragraphs": 1, "prediction_paragraphs": 1},
            {"sample_id": "2", "lang": "de-en", "source": "Hallo", "target": "Hello", "prediction": "",
             "generation_limit_reached": False, "source_paragraphs": 1, "prediction_paragraphs": 0},
        ]
        args = SimpleNamespace(wmt23_metric="comet22", wmt23_batch_size=1, wmt23_max_new_tokens=512, split="test")
        dataset = SimpleNamespace(all_data={"en-de": [], "de-en": []}, dataset_metadata={"test": "fixture"})
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(mt_evaluate.WMT_DATASETS, {"wmt23": lambda *a, **kw: dataset}), \
                patch.object(mt_evaluate, "build_wmt_evaluation_samples", return_value=rows), \
                patch.object(mt_evaluate, "generate_wmt_predictions", return_value=rows), \
                patch.object(mt_evaluate, "score_comet22", return_value=([0.9, 0.1], {"model_id": "fixture"})) as score:
            root = Path(temporary)
            result = mt_evaluate.evaluate_wmt_scope(args, None, None, None, "in", root, task="wmt23")
            score.assert_called_once_with(rows, args)
            self.assertEqual(result["by_language"]["en-de"]["comet22_x100"], 90)
            self.assertEqual(result["by_language"]["de-en"]["comet22_x100"], 10)
            self.assertEqual(result["macro_average"]["comet22"], 0.5)
            self.assertEqual(json.loads((root / "wmt23_metrics.json").read_text())["status"], "scored")
            self.assertTrue((root / "wmt23_predictions.de-en.jsonl").is_file())

    def test_missing_comet_interpreter_fails_before_loading_translation_model(self):
        args = SimpleNamespace(eval_sample_log_limit=0, tasks=["wmt23"], wmt23_metric="comet22", comet22_python=None)
        with self.assertRaisesRegex(ValueError, "comet22_python"):
            mt_evaluate.validate_eval_args(args)


if __name__ == "__main__":
    unittest.main()
