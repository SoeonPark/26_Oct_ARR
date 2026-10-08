"""CPU checks for distance variance, loss selection, and detached diagnostics."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from config import ALIGNMENT_LOSSES, parse_args, resolve_alignment_loss, validate_gap_config
from models import CustomModel


class TinyEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor([[1., 0.], [0., 1.], [1., 1.],
                          [4., 4.], [0., 2.], [-1., -1.]]),
            freeze=False,
        )

    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(hidden_states=(hidden, 2 * hidden))


def make_model(loss_type=None):
    config = SimpleNamespace(
        alignment_hidden_state_layer=0,
        alignment_hidden_state_position="last_token",
        alignment_temperature=0.2,
    )
    if loss_type is not None:
        config.alignment_loss = loss_type
    return CustomModel(config, TinyEmbeddingModel())


def alignment_batch():
    return {
        "source_input_ids": torch.tensor([[0], [1], [2]]),
        "target_input_ids": torch.tensor([[3], [4], [5]]),
        "source_attention_mask": torch.ones(3, 1, dtype=torch.long),
        "target_attention_mask": torch.ones(3, 1, dtype=torch.long),
        "lang_pair": ["en-ko"] * 3,
    }


class GapLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_equal_distances_opposite_directions_have_zero_loss(self):
        source = torch.zeros(2, 2, requires_grad=True)
        target = torch.tensor([[3., 4.], [-3., -4.]], requires_grad=True)
        loss, values = make_model().compute_gap_consistency_loss(
            source, target, ["en-ko"] * 2, return_per_sample=True,
        )
        self.assertEqual(loss.item(), 0.)
        torch.testing.assert_close(values["gap_distance"], torch.tensor([5., 5.]))
        self.assertEqual(values["gap_distance_mean"].item(), 5.)
        loss.backward()
        torch.testing.assert_close(target.grad, torch.zeros_like(target))

    def test_alignment_logit_slicing_preserves_embeddings_loss_and_gradients(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(7)
        base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                          num_hidden_layers=2, num_attention_heads=2,
                                          num_key_value_heads=2, use_cache=False))
        original_forward = base.forward
        batch = {
            "source_input_ids": torch.tensor([[1, 2, 3], [4, 5, 0], [6, 7, 8]]),
            "target_input_ids": torch.tensor([[9, 10, 11], [12, 13, 0], [14, 15, 16]]),
            "source_attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]]),
            "target_attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]]),
            "lang_pair": ["cs-en"] * 3,
        }
        for position, layer in (("last_token", -1), ("mean", 1)):
            config = SimpleNamespace(alignment_loss="centered_infonce", alignment_temperature=0.05,
                                     alignment_hidden_state_position=position, alignment_hidden_state_layer=layer)
            model = CustomModel(config, base).eval()
            base.zero_grad(set_to_none=True)
            optimized = model(alignment=batch)
            optimized["loss"].backward()
            gradients = {name: p.grad.clone() for name, p in base.named_parameters() if p.grad is not None}
            base.zero_grad(set_to_none=True)

            def full_logits(**kwargs):
                kwargs["logits_to_keep"] = 0
                return original_forward(**kwargs)

            with patch.object(base, "forward", side_effect=full_logits):
                reference = model(alignment=batch)
            reference["loss"].backward()
            for key in ("loss", "source_embeddings", "target_embeddings"):
                torch.testing.assert_close(optimized[key], reference[key])
            for name, parameter in base.named_parameters():
                if name in gradients:
                    torch.testing.assert_close(gradients[name], parameter.grad)

    def test_population_variance_of_raw_distances(self):
        source = torch.zeros(2, 2, requires_grad=True)
        target = torch.tensor([[3., 4.], [0., 1.]], requires_grad=True)
        loss, values = make_model().compute_gap_consistency_loss(
            source, target, ["en-ko"] * 2, return_per_sample=True,
        )
        self.assertEqual(loss.item(), 4.)
        torch.testing.assert_close(values["gap_distance"], torch.tensor([5., 1.]))
        self.assertEqual(values["gap_distance_mean"].item(), 3.)
        torch.testing.assert_close(values["per_sample_loss"], torch.tensor([4., 4.]))
        loss.backward()
        torch.testing.assert_close(target.grad, torch.tensor([[1.2, 1.6], [0., -2.]]))
        torch.testing.assert_close(source.grad, -target.grad)

    def test_double_precision_gradcheck(self):
        source = torch.tensor([[1., 2.], [3., 5.]], dtype=torch.float64,
                              requires_grad=True)
        target = torch.tensor([[2., 6.], [7., 3.]], dtype=torch.float64,
                              requires_grad=True)
        model = make_model()
        self.assertTrue(torch.autograd.gradcheck(
            lambda src, tgt: model.compute_gap_consistency_loss(src, tgt, ["en-ko"] * 2),
            (source, target),
        ))
        loss, values = model.compute_gap_consistency_loss(
            source, target, ["en-ko"] * 2, return_per_sample=True,
        )
        self.assertEqual(loss.dtype, torch.float64)
        self.assertEqual(values["gap_distance"].dtype, torch.float64)

    def test_zero_distances_have_finite_backward(self):
        # A norm at zero is nondifferentiable; PyTorch selects a zero
        # subgradient. Require finite backward here, not numerical gradcheck.
        for target_values in ([[0., 0.], [0., 0.]], [[0., 0.], [3., 4.]]):
            with self.subTest(target=target_values):
                source = torch.zeros(2, 2, requires_grad=True)
                target = torch.tensor(target_values, requires_grad=True)
                loss, values = make_model().compute_gap_consistency_loss(
                    source, target, ["en-ko"] * 2, return_per_sample=True,
                )
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(source.grad).all())
                self.assertTrue(torch.isfinite(target.grad).all())
                for value in values.values():
                    self.assertTrue(torch.isfinite(value).all())

    def test_low_precision_upcasts_before_subtraction_and_preserves_gradient(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                source = torch.tensor([[50000., 0.], [0., 0.]], dtype=dtype,
                                      requires_grad=True)
                target = torch.tensor([[-50000., 0.], [100., 0.]], dtype=dtype,
                                      requires_grad=True)
                loss, values = make_model().compute_gap_consistency_loss(
                    source, target, ["en-ko"] * 2, return_per_sample=True,
                )
                reference = (target.float() - source.float()).norm(dim=-1).var(correction=0)
                torch.testing.assert_close(loss, reference)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertEqual(values["gap_distance"].dtype, torch.float32)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertEqual(source.grad.dtype, dtype)
                self.assertTrue(torch.isfinite(source.grad).all())
                self.assertTrue(torch.isfinite(target.grad).all())

    def test_invalid_gap_batches_fail(self):
        valid = torch.zeros(2, 3)
        cases = (
            (valid[0], valid[0], ["en-ko"]),
            (valid, torch.zeros(2, 4), ["en-ko"] * 2),
            (valid[:0], valid[:0], []),
            (valid[:1], valid[:1], ["en-ko"]),
            (valid, valid, None),
            (valid, valid, ["en-ko"]),
            (valid, valid, ["en-ko", "en-ja"]),
            (valid, valid, ["en-ko", "ko-en"]),
        )
        for source, target, pairs in cases:
            with self.subTest(shape=source.shape, pairs=pairs), self.assertRaises(AssertionError):
                make_model().compute_gap_consistency_loss(source, target, pairs)

    def test_forward_routes_selected_loss_and_updates_parameters(self):
        for loss_type in (None, "infonce", "gap_consistency"):
            with self.subTest(loss_type=loss_type):
                model = make_model(loss_type)
                output = model(alignment=alignment_batch(), return_per_sample=True)
                source, target = output["source_embeddings"], output["target_embeddings"]
                if loss_type == "gap_consistency":
                    expected = (target - source).norm(dim=-1).var(correction=0)
                else:
                    logits = F.normalize(source, dim=-1) @ F.normalize(target, dim=-1).T / .2
                    labels = torch.arange(len(source))
                    expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
                self.assertEqual(output["alignment_loss_type"], loss_type or "infonce")
                torch.testing.assert_close(output["loss"], expected)
                self.assertTrue(output["loss"].requires_grad)
                optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                before = model.basemodel.embedding.weight.detach().clone()
                output["loss"].backward()
                optimizer.step()
                self.assertFalse(torch.equal(before, model.basemodel.embedding.weight))

    def test_diagnostics_are_detached_and_do_not_change_loss_or_gradient(self):
        for loss_type in ALIGNMENT_LOSSES:
            with self.subTest(loss_type=loss_type):
                model = make_model(loss_type)
                plain = model(alignment=alignment_batch())
                plain["loss"].backward()
                reference_grad = model.basemodel.embedding.weight.grad.detach().clone()
                model.zero_grad()
                recorded = model(alignment=alignment_batch(), return_per_sample=True)
                recorded["loss"].backward()
                torch.testing.assert_close(recorded["loss"], plain["loss"])
                torch.testing.assert_close(model.basemodel.embedding.weight.grad, reference_grad)
                for name in ("per_sample_loss", "gap_distance", "source_norm",
                             "target_norm", "positive_cosine", "gap_distance_mean"):
                    value = recorded[name]
                    self.assertFalse(value.requires_grad)
                    self.assertEqual(value.shape, torch.Size([] if name == "gap_distance_mean" else [3]))
                torch.testing.assert_close(recorded["per_sample_loss"].mean(), recorded["loss"].detach())


class ContrastiveVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_forward_matches_pairwise_formula_and_keeps_raw_embeddings(self):
        for method in ALIGNMENT_LOSSES[2:]:
            with self.subTest(method=method):
                model = make_model(method)
                model.experiment_config.alignment_gap_scale = 2.5
                output = model(alignment=alignment_batch(), return_per_sample=True)
                a, b = output["source_embeddings"], output["target_embeddings"]
                mean_distance = torch.stack([(y-x).norm() for x, y in zip(a, b)]).mean()
                mean_gap = b.mean(0) - a.mean(0)
                rows = []
                for x in a:
                    row = []
                    for y in b:
                        if method == "gap_distance_infonce":
                            score = -(((y-x).norm()-mean_distance)/2.5).square()
                        elif method == "centered_infonce":
                            score = F.cosine_similarity(x-a.mean(0), y-b.mean(0), dim=0)
                        else:
                            score = F.cosine_similarity(y-x, mean_gap, dim=0)
                        row.append(score)
                    rows.append(torch.stack(row))
                z = torch.stack(rows) / .2
                expected = ((z.logsumexp(1)-z.diagonal()) + (z.logsumexp(0)-z.diagonal())) / 2
                self.assertEqual(output["alignment_loss_type"], method)
                torch.testing.assert_close(output["per_sample_loss"], expected.detach())
                torch.testing.assert_close(output["loss"], expected.mean())
                torch.testing.assert_close(a, model.basemodel.embedding(alignment_batch()["source_input_ids"])[:, 0])
                torch.testing.assert_close(b, model.basemodel.embedding(alignment_batch()["target_input_ids"])[:, 0])

    def test_reference_gradients_and_low_precision_autocast(self):
        generator = torch.Generator().manual_seed(19)
        for method in ALIGNMENT_LOSSES[2:]:
            with self.subTest(method=method):
                model = make_model(method)
                a = torch.randn(3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
                b = torch.randn(3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
                fn = lambda x, y: model.compute_contrastive_variant_loss(x, y, ["en-ko"]*3)
                self.assertTrue(torch.autograd.gradcheck(fn, (a, b)))
                for dtype in (torch.float16, torch.bfloat16):
                    x, y = a.detach().to(dtype), b.detach().to(dtype)
                    expected = fn(x.float(), y.float())
                    x.requires_grad_(); y.requires_grad_()
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        actual = fn(x, y)
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected)
                    actual.backward()
                    self.assertTrue(torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all())

    def test_zero_gaps_and_zero_mean_gap_have_finite_backward(self):
        for method in ALIGNMENT_LOSSES[2:]:
            for values in (([[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]),
                           ([[-1., 0.], [1., 0.]], [[1., 0.], [-1., 0.]])):
                with self.subTest(method=method, values=values):
                    a, b = [torch.tensor(v, requires_grad=True) for v in values]
                    loss = make_model(method).compute_contrastive_variant_loss(a, b, ["en-ko"]*2)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())
                    if not torch.count_nonzero(a):
                        torch.testing.assert_close(loss, torch.tensor(2.).log())

    def test_cli_and_pair_batch_validation(self):
        for method in ALIGNMENT_LOSSES[2:]:
            with self.subTest(method=method):
                with patch("sys.argv", ["test", "--alignment_loss", method, "--training_type",
                                        "contrastive_only", "--alignment_batching", "same_pair",
                                        "--alignment_gap_scale", "2.5"]):
                    args = parse_args()
                validate_gap_config(args)
                self.assertEqual(args.alignment_gap_scale, 2.5)
                for update in (dict(alignment_batching="mixed"), dict(batch_size=1),
                               dict(eval_batch_size=1), dict(alignment_temperature=0)):
                    with self.assertRaises(ValueError):
                        validate_gap_config(SimpleNamespace(**(vars(args) | update)))
                with self.assertRaises(AssertionError):
                    make_model(method).compute_contrastive_variant_loss(
                        torch.ones(2, 3), torch.ones(2, 3), ["en-ko", "en-ja"],
                    )


class GapConfigTests(unittest.TestCase):
    def test_legacy_configuration_defaults_and_invalid_loss(self):
        for config in (None, SimpleNamespace(), {}):
            self.assertEqual(resolve_alignment_loss(config), "infonce")
        for config in (SimpleNamespace(alignment_loss="gap_consistency"),
                       {"alignment_loss": "gap_consistency"}):
            self.assertEqual(resolve_alignment_loss(config), "gap_consistency")
        with self.assertRaises(ValueError):
            resolve_alignment_loss(SimpleNamespace(alignment_loss="unknown"))
        with self.assertRaises(ValueError):
            make_model("unknown")(alignment=alignment_batch())

    def test_cli_defaults_and_gap_options(self):
        with patch("sys.argv", ["test"]):
            defaults = parse_args()
        self.assertEqual(defaults.alignment_loss, "infonce")
        self.assertEqual(defaults.train_sample_log_interval, 1000)
        self.assertEqual(defaults.train_sample_log_limit, 8)
        validate_gap_config(defaults)
        with patch("sys.argv", ["test", "--alignment_loss", "gap_consistency",
                                "--training_type", "contrastive_only",
                                "--alignment_batching", "same_pair",
                                "--train_sample_log_interval", "0",
                                "--train_sample_log_limit", "0"]):
            selected = parse_args()
        validate_gap_config(selected)
        self.assertEqual(selected.alignment_loss, "gap_consistency")

    def test_gap_config_rejects_invalid_batching_and_sizes(self):
        base = dict(alignment_loss="gap_consistency", training_type="contrastive_only",
                    alignment_batching="same_pair", batch_size=2, eval_batch_size=2)
        validate_gap_config(SimpleNamespace(**base))
        for update in (dict(alignment_batching="mixed"), dict(batch_size=1), dict(eval_batch_size=1)):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_gap_config(SimpleNamespace(**(base | update)))
        validate_gap_config(SimpleNamespace(**(base | dict(
            training_type="transfer_only", alignment_batching="mixed", batch_size=1,
        ))))
        for training_type in ("alternative", "contrastive_then_transfer"):
            with self.subTest(training_type=training_type), self.assertRaises(ValueError):
                validate_gap_config(SimpleNamespace(**(base | dict(
                    training_type=training_type, alignment_batching="mixed",
                ))))

    def test_negative_log_settings_fail_for_both_losses(self):
        for loss_type in ("infonce", "gap_consistency"):
            for name in ("train_sample_log_interval", "train_sample_log_limit", "eval_sample_log_limit"):
                with self.subTest(loss_type=loss_type, name=name), self.assertRaises(ValueError):
                    validate_gap_config(SimpleNamespace(alignment_loss=loss_type, **{name: -1}))


if __name__ == "__main__":
    unittest.main()
