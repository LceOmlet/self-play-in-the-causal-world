"""Regression: certified truth must itself be an admissible terminal answer."""
import json
from fractions import Fraction

import pytest

from cpt_world import CPTWorldEnvironment, compute_query_truth
from cpt_world.query_truth import CounterfactualIntervalCertificate
from cpt_world.rewards import terminal_quality_reward
from cpt_world.task_scoring import score_terminal_answer


@pytest.mark.parametrize("sample_index,anchor_index", [(200044, 19), (200059, 0)])
def test_real_counterfactual_truth_is_a_legal_perfect_answer(sample_index, anchor_index):
    env = CPTWorldEnvironment()
    env.reset(prompt=[], sample_index=sample_index, anchor_index=anchor_index,
              query_type="individual_counterfactual_probability", tape_key="boundary-regression")
    world, seed = env.episode.world, env.episode.seed
    truth = compute_query_truth(world, seed)
    assert 0 <= truth["lower"] <= truth["upper"] <= 1
    answer = json.dumps({"type": "answer", "lower": truth["lower"], "upper": truth["upper"]})
    diagnostic = score_terminal_answer(answer, seed, world, terminal_truth=truth)
    assert terminal_quality_reward(diagnostic) == 1


@pytest.mark.parametrize("lower,upper", [(0.0, 1.01), (-0.01, 1.0), (0.8, 0.2),
                                       (float("nan"), 1.0), (0.0, float("inf")),
                                       (Fraction(0), Fraction(10**15 + 1, 10**15))])
def test_inconsistent_certificates_are_not_silently_clipped(lower, upper):
    with pytest.raises(ValueError):
        CounterfactualIntervalCertificate(lower, upper, "exact", 0.0)


def test_exact_rational_certificate_preserves_exact_arithmetic():
    cert = CounterfactualIntervalCertificate(Fraction(1, 7), Fraction(5, 7), "exact", 0.0)
    assert isinstance(cert.lower, Fraction) and isinstance(cert.upper, Fraction)
    assert cert.lower == Fraction(1, 7) and cert.upper == Fraction(5, 7)
