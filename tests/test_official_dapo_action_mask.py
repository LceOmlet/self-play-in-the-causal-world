"""Exercise the real recipe method's multi-turn action-mask contract."""

import unittest
from types import SimpleNamespace

import torch
from dapo.dapo_ray_trainer import RayDAPOTrainer
from verl import DataProto


class OfficialDAPOActionMaskTests(unittest.TestCase):
    def batch(self, with_mask=True):
        tensors = {
            "responses": torch.arange(12).reshape(2, 6),
            "attention_mask": torch.tensor([[1] * 9, [1] * 7 + [0, 0]]),
        }
        if with_mask:
            tensors["response_mask"] = torch.tensor([[1, 0, 0, 1, 1, 1], [1, 0, 1, 1, 0, 0]])
        return DataProto.from_dict(tensors=tensors)

    def owner(self, seen):
        def old_log_probability(batch):
            seen.append(batch.batch["response_mask"].clone())
            return DataProto.from_dict(
                tensors={"old_log_probs": torch.full((2, 6), -2.0), "entropys": torch.ones(2, 6)}
            ), 0.0

        return SimpleNamespace(
            _compute_old_log_prob=old_log_probability,
            use_reference_policy=False,
            config=SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(
                    actor=SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None)
                )
            ),
        )

    def test_existing_tool_action_mask_reaches_probability_owner_and_survives_union(self):
        batch = self.batch()
        expected = batch.batch["response_mask"].clone()
        seen = []
        output = RayDAPOTrainer.compute_kl_related_metrics(self.owner(seen), batch, {}, {})
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0], expected))
        self.assertTrue(torch.equal(output.batch["response_mask"], expected))
        self.assertIn("old_log_probs", output.batch)

    def test_single_turn_missing_mask_uses_official_attention_fallback(self):
        batch = self.batch(with_mask=False)
        expected = batch.batch["attention_mask"][:, -6:].clone()
        seen = []
        output = RayDAPOTrainer.compute_kl_related_metrics(self.owner(seen), batch, {}, {})
        self.assertTrue(torch.equal(output.batch["response_mask"], expected))
        self.assertTrue(torch.equal(seen[0], expected))

    def test_existing_all_zero_action_mask_is_not_replaced(self):
        batch = self.batch()
        batch.batch["response_mask"].zero_()
        seen = []
        output = RayDAPOTrainer.compute_kl_related_metrics(self.owner(seen), batch, {}, {})
        self.assertEqual(output.batch["response_mask"].sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
