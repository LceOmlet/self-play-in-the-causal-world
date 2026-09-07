"""Compare the real fused/unfused trainers against the standard DAPO equation."""

from __future__ import annotations

import unittest
from collections import defaultdict
from types import SimpleNamespace

import torch
from trl import GRPOTrainer


def paper_dapo_loss(logps, old_logps, advantages, mask, denominator, correction=None):
    """DAPO paper Eq. 8; optional existing sampler correction is kept explicit."""
    ratio = (logps - old_logps).exp()
    unclipped = ratio * advantages[:, None]
    clipped = ratio.clamp(0.8, 1.28) * advantages[:, None]
    per_token = -torch.minimum(unclipped, clipped)
    if correction is not None:
        per_token = per_token * correction
    return (per_token * mask).sum() / denominator


def fake_trainer(model, accumulation, generation_steps, correction, training=True):
    trainer = object.__new__(GRPOTrainer)
    trainer.model = model
    model.train(training)
    trainer.loss_type = "dapo"
    trainer.beta = 0.0
    trainer.epsilon_low, trainer.epsilon_high = 0.2, 0.28
    trainer.importance_sampling_level = "token"
    trainer.top_entropy_quantile = 1.0
    trainer.off_policy_mask_threshold = None
    trainer.aux_loss_enabled = False
    trainer._entropy_bonus_enabled = False
    trainer.use_vllm = correction
    trainer.vllm_importance_sampling_correction = correction
    trainer.current_gradient_accumulation_steps = accumulation
    trainer.args = SimpleNamespace(
        steps_per_generation=generation_steps,
        delta=None,
        use_bias_correction_kl=False,
    )
    trainer.accelerator = SimpleNamespace(
        num_processes=1,
        gather=lambda x: x,
        reduce=lambda x, **kwargs: x,
    )
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    return trainer


class DAPOEquationTests(unittest.TestCase):
    def test_clip_higher_has_standard_asymmetric_gradient(self):
        ratios = torch.tensor([0.7, 0.9, 1.1, 1.25, 1.4], dtype=torch.float64)
        logps = ratios.log().repeat(2, 1).requires_grad_()
        advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
        loss = paper_dapo_loss(
            logps,
            torch.zeros_like(logps),
            advantages,
            torch.ones_like(logps),
            10,
        )
        gradient = torch.autograd.grad(loss, logps)[0]
        torch.testing.assert_close(
            gradient,
            torch.tensor(
                [[-0.07, -0.09, -0.11, -0.125, 0.0], [0.0, 0.09, 0.11, 0.125, 0.14]],
                dtype=torch.float64,
            ),
        )





@unittest.skipUnless(torch.cuda.is_available(), "Fused comparison requires CUDA")
class DAPOFusedComparisonTests(unittest.TestCase):
    def test_fused_unfused_and_paper_gradients_with_accumulation_and_tools(self):
        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

        torch.manual_seed(31)
        device = "cuda"
        h0 = torch.randn(4, 8, 16, device=device) * 0.2
        w0 = torch.randn(32, 16, device=device) * 0.2
        targets = torch.randint(0, 32, (4, 8), device=device)
        completion_mask = (
            torch.arange(8, device=device)[None, :]
            < torch.tensor(
                [2, 4, 6, 8],
                device=device,
            )[:, None]
        )
        tool_mask = torch.ones_like(completion_mask)
        tool_mask[2, 2:4] = False
        mask = completion_mask & tool_mask
        advantages = torch.tensor([-1.0, -0.3, 0.4, 0.9], device=device)
        initial_logps = (h0 @ w0.T).log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
        # Cross both clipping boundaries, on both positive and negative advantages.
        old_logps = initial_logps - torch.linspace(-0.5, 0.5, 8, device=device)[None, :]
        for accumulation, generation_steps, training in ((4, 4, True), (2, 4, True), (4, 4, False)):
            for corrected_sampling in (False, True):
                with self.subTest(
                    accumulation=accumulation, train=training, correction=corrected_sampling
                ):
                    correction = (
                        torch.tensor([0.0, 0.3, 1.0, 2.0], device=device)[:, None]
                        if corrected_sampling
                        else None
                    )
                    results = {}
                    for backend in ("fused", "unfused", "paper"):
                        h = h0.clone().requires_grad_()
                        model = torch.nn.Module()
                        model.lm_head = torch.nn.Linear(16, 32, bias=False, device=device)
                        model.lm_head.weight.data.copy_(w0)
                        trainer = fake_trainer(
                            model, accumulation, generation_steps, corrected_sampling, training
                        )
                        if backend == "fused":
                            trainer.liger_loss = LigerFusedLinearGRPOLoss(
                                beta=0.0,
                                loss_type="dapo",
                                use_ref_model=False,
                                epsilon_low=0.2,
                                epsilon_high=0.28,
                            )
                        total_loss = 0.0
                        for i in range(4):
                            logits = (h[i : i + 1] @ model.lm_head.weight.T).log_softmax(-1)
                            logps = logits.gather(-1, targets[i : i + 1, :, None]).squeeze(-1)
                            trainer._get_last_hidden_state = lambda *args, index=i, values=h: (
                                values[index : index + 1]
                            )
                            trainer._get_per_token_logps_and_entropies = (
                                lambda *args, values=logps, **kwargs: (
                                    values,
                                    torch.zeros_like(values),
                                    None,
                                )
                            )
                            inputs = {
                                "prompt_ids": targets[i : i + 1, :1],
                                "prompt_mask": torch.ones_like(targets[i : i + 1, :1]),
                                "completion_ids": targets[i : i + 1],
                                "completion_mask": completion_mask[i : i + 1],
                                "tool_mask": tool_mask[i : i + 1],
                                "advantages": advantages[i : i + 1],
                                "old_per_token_logps": old_logps[i : i + 1],
                                "num_items_in_batch": mask.sum(),
                            }
                            if correction is not None:
                                inputs["importance_sampling_ratio"] = correction[i : i + 1]
                            if backend == "fused":
                                loss = trainer.compute_liger_loss(model, inputs)
                            elif backend == "unfused":
                                loss = GRPOTrainer._compute_loss(trainer, model, inputs)
                            else:
                                factor = accumulation / generation_steps if training else 1
                                loss = paper_dapo_loss(
                                    logps,
                                    old_logps[i : i + 1],
                                    advantages[i : i + 1],
                                    mask[i : i + 1],
                                    mask.sum() * factor,
                                    correction[i : i + 1] if correction is not None else None,
                                )
                            total_loss += loss.detach().item()
                            loss.backward()
                        results[backend] = (
                            torch.tensor(total_loss),
                            h.grad,
                            model.lm_head.weight.grad,
                        )
                    for backend in ("fused", "unfused"):
                        for actual, expected in zip(
                            results[backend], results["paper"], strict=True
                        ):
                            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
                    self.assertTrue(
                        torch.equal(
                            results["fused"][1][~mask], torch.zeros_like(results["fused"][1][~mask])
                        )
                    )


if __name__ == "__main__":
    unittest.main()
