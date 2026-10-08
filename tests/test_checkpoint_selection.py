"""Checkpoint selection uses the declared budget and the actual saved step."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.select_checkpoint import main


class CheckpointSelectionTests(unittest.TestCase):
    def make_run(self, root, *, actual_step=100, budget=100, saved_step=100, loss="gap_consistency"):
        run = Path(root)
        config = {"checkpoint_global_step": saved_step}
        if budget is not None:
            config["num_steps"] = budget
        if loss is not None:
            config["alignment_loss"] = loss
        (run / "experiment_config.json").write_text(json.dumps(config))
        (run / "trainer_state.json").write_text(json.dumps({
            "global_step": actual_step,
            "log_history": [{"step": actual_step, "eval_align_in_en-ko_loss": 0.5}],
        }))
        (run / "adapter_model.safetensors").touch()
        return run

    def select(self, run, rule="final_step"):
        output = io.StringIO()
        with patch("sys.argv", ["select_checkpoint.py", str(run), "--rule", rule]), redirect_stdout(output):
            main()
        return output.getvalue()

    def test_completed_budget_selects_matching_root_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp)
            output = self.select(run)
            self.assertIn("Best step  : 100", output)
            self.assertIn(f"Checkpoint : {run}", output)
            self.assertIn("python3 evaluate.py", output)

    def test_interrupted_run_is_not_a_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, actual_step=50, saved_step=50)
            with self.assertRaisesRegex(SystemExit, "configured 100 steps.*records 50"):
                self.select(run)

    def test_missing_budget_requires_saved_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, budget=None)
            with self.assertRaisesRegex(SystemExit, "positive num_steps"):
                self.select(run)

    def test_budget_can_come_from_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, budget=None)
            (run / "run_metadata.json").write_text(json.dumps({
                "experiment_config": {"num_steps": 100},
            }))
            self.assertIn("Best step  : 100", self.select(run))

    def test_stale_or_unidentified_root_adapter_is_not_reused(self):
        for saved_step in (50, None):
            with self.subTest(saved_step=saved_step), tempfile.TemporaryDirectory() as tmp:
                run = self.make_run(tmp, saved_step=saved_step)
                output = self.select(run)
                self.assertIn("Checkpoint : MISSING", output)
                self.assertNotIn("python3 evaluate.py", output)
                self.assertIn("checkpoint_global_step=100", output)

    def test_numbered_checkpoint_is_preferred_to_stale_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, saved_step=50)
            checkpoint = run / "checkpoint-100"
            checkpoint.mkdir()
            output = self.select(run)
            self.assertIn(f"Checkpoint : {checkpoint}", output)
            self.assertNotIn("MISSING", output)

    def test_old_loss_configuration_defaults_to_infonce(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, loss=None)
            output = self.select(run)
            self.assertIn("Alignment loss: infonce", output)
            self.assertIn("Best step  : 100", output)

    def test_wmt_default_uses_seen_validation_and_only_saved_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp, actual_step=5000, budget=100000, saved_step=None, loss="centered_infonce")
            (run / "experiment_config.json").write_text(json.dumps({
                "downstream_task": "wmt23", "training_type": "alternative",
                "alignment_loss": "centered_infonce", "num_steps": 100000,
            }))
            (run / "trainer_state.json").write_text(json.dumps({
                "global_step": 5000,
                "log_history": [{"step": step, "eval_wmt23_in_de_loss": loss,
                                 "eval_align_out_en-zh_loss": 0.01}
                                for step, loss in [(0, 0.1), (2500, 0.2), (5000, 0.3)]],
            }))
            (run / "checkpoint-5000").mkdir()
            output = io.StringIO()
            with patch("sys.argv", ["select_checkpoint.py", str(run)]), redirect_stdout(output):
                main()
            self.assertIn("Rule  : wmt23_in", output.getvalue())
            self.assertIn("Best step  : 5000", output.getvalue())
            self.assertIn("Excluded unsaved validation steps: [0, 2500]", output.getvalue())
            self.assertIn("--tasks alignment wmt23", output.getvalue())

    def test_task_validation_can_select_matching_final_root_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp)
            output = self.select(run, "align_in")
            self.assertIn(f"Checkpoint : {run}", output)
            self.assertNotIn("MISSING", output)


if __name__ == "__main__":
    unittest.main()
