import torch

from src.rl.grpo import (
    compute_group_advantages,
    gaussian_logprob,
    grpo_clipped_loss,
    sample_gaussian_transition,
    scheduler_transition_mean,
)
from src.utils.scheduler import FlowMatchScheduler


def test_scheduler_transition_mean_matches_step_exactly():
    scheduler = FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(8)
    sample = torch.randn(1, 3, 2, 4, 1)
    model_output = torch.randn_like(sample)
    timestep = scheduler.timesteps[3]

    expected = scheduler.step(model_output, timestep, sample)
    actual = scheduler_transition_mean(scheduler, model_output, timestep, sample)

    assert torch.equal(actual, expected)


def test_seeded_gaussian_transition_and_logprob_are_reproducible():
    mean = torch.zeros(1, 2, 3, 4, 1)
    mask = torch.ones_like(mean, dtype=torch.bool)

    gen_a = torch.Generator(device="cpu").manual_seed(123)
    gen_b = torch.Generator(device="cpu").manual_seed(123)
    sample_a = sample_gaussian_transition(mean, 0.1, generator=gen_a)
    sample_b = sample_gaussian_transition(mean, 0.1, generator=gen_b)

    assert torch.equal(sample_a, sample_b)
    logprob_a = gaussian_logprob(sample_a, mean, 0.1, mask=mask)
    logprob_b = gaussian_logprob(sample_b, mean, 0.1, mask=mask)
    assert torch.allclose(logprob_a, logprob_b)


def test_group_advantages_zero_mean_for_same_group():
    adv = compute_group_advantages([1.0, 0.0, 1.0, 0.0])
    assert torch.allclose(adv.mean(), torch.tensor(0.0), atol=1e-6)


def test_grpo_clipped_loss_returns_finite_stats():
    new_logprob = torch.tensor([0.0, -0.2, 0.1])
    old_logprob = torch.tensor([-0.1, -0.1, 0.0])
    advantages = compute_group_advantages([1.0, 0.0, 1.0])

    stats = grpo_clipped_loss(new_logprob, old_logprob, advantages, clip_range=0.2)

    assert torch.isfinite(stats.loss)
    assert torch.isfinite(stats.ratio_mean)
    assert torch.isfinite(stats.clipfrac)
    assert torch.isfinite(stats.approx_kl)
