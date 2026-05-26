import torch
from torch import nn

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


def _make_fake_grpo_server(*, noise_schedule="per_step"):
    # GRPOTrainingServer.__init__ pulls in the full inference stack (T5, VAE,
    # transformer weights, distributed init, ...). For the eval_mode sampling
    # branch we only need the subset of attributes _sample_action_chunk reads;
    # __new__ skips the heavy init and we patch the transformer / video-prefix
    # calls with shape-correct stubs.
    from src.rl.server import GRPOTrainingServer

    server = GRPOTrainingServer.__new__(GRPOTrainingServer)
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.latent_height = 4
    server.latent_width = 4
    server.action_per_frame = 2
    server.noise_schedule = noise_schedule
    server.normalize_denoising_horizon = True
    server.normalize_action_dim = True
    server.action_noise_std = 0.05
    server.frame_st_id = 0
    server.cache_name = "test"
    server.action_mask = torch.ones(30, dtype=torch.bool)

    class _Cfg:
        frame_chunk_size = 2
        action_dim = 30
        action_num_inference_steps = 3

    server.job_config = _Cfg()
    server.action_scheduler = FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True)

    server._run_video_prefix = lambda *args, **kwargs: None

    def _stub_transformer_step(actions, t, frame_st_id, *, last_step):
        cond = (
            torch.zeros(
                [1, server.job_config.action_dim, 1, server.action_per_frame, 1],
                device=server.device,
                dtype=server.dtype,
            )
            if frame_st_id == 0
            else None
        )
        if last_step:
            return None, cond
        return torch.zeros_like(actions), cond

    server._action_transformer_step = _stub_transformer_step

    def _stub_logprob_mask(frame_st_id, reference):
        mask = torch.ones_like(reference, dtype=torch.bool)
        if frame_st_id == 0:
            mask[:, :, 0:1] = False
        return mask

    server._action_logprob_mask = _stub_logprob_mask
    server.postprocess_action = lambda actions: actions.detach().cpu().numpy()
    return server


def test_sample_action_chunk_eval_mode_returns_placeholder_chunk():
    server = _make_fake_grpo_server(noise_schedule="per_step")

    _, chunk = server._sample_action_chunk({"obs": None}, frame_st_id=0, eval_mode=True)

    assert chunk.action_chain == []
    assert torch.equal(chunk.old_logprobs, torch.zeros(1))
    assert chunk.action_timesteps.numel() == 0


def test_sample_action_chunk_training_mode_per_step_records_logprobs():
    server = _make_fake_grpo_server(noise_schedule="per_step")

    _, chunk = server._sample_action_chunk({"obs": None}, frame_st_id=0, eval_mode=False)

    # per_step records (initial + one append per noise-bearing step) so
    # action_chain is non-empty and old_logprobs / action_timesteps carry
    # real values — guards against the eval_mode branch leaking into training.
    assert len(chunk.action_chain) >= 2
    assert chunk.old_logprobs.numel() > 0
    assert chunk.action_timesteps.numel() > 0


class _SyntheticChunkPolicy(nn.Module):
    """Trainable map (chunk_input -> Gaussian mean) shaped like the action transformer.

    Stand-in for the WAN transformer + scheduler.step in tests that only need
    to verify gradient direction through the per-chunk forward + gaussian_logprob
    path. Output shape (B, C, F, N, 1) matches what gaussian_logprob receives in
    the real server.
    """

    def __init__(self, in_dim: int, out_shape: tuple[int, ...]):
        super().__init__()
        self.out_shape = out_shape
        out_elems = 1
        for d in out_shape:
            out_elems *= d
        self.linear = nn.Linear(in_dim, out_elems)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).reshape(self.out_shape)


def _episode_logprob_for_test(policy, chunks, std, *, normalize_action_dim, normalize_episode_length):
    """Mirror server.get_action_logprobs: per-chunk forward + sum/mean across chunks."""
    chunk_lps = []
    for chunk in chunks:
        mean = policy(chunk["input"])
        lp = gaussian_logprob(
            chunk["action"], mean, std,
            mask=chunk["mask"],
            normalize_action_dim=normalize_action_dim,
        ).reshape(())
        chunk_lps.append(lp)
    summed = torch.stack(chunk_lps).sum()
    if normalize_episode_length:
        summed = summed / len(chunks)
    return summed


def _backward_episode_for_test(
    policy, chunks, grad_coef, std, *, normalize_action_dim, normalize_episode_length,
):
    """Mirror _backward_episode_logprob: per-chunk grad-forward + backward(chunk_lp * coef/n_chunks).

    The real server accumulates gradients into the model params via per-chunk
    .backward() (the only autograd surface, after the no_grad detached forward
    has set ratio + clip mask). This helper reproduces that exact accumulation
    pattern so the test verifies the same code path's sign behavior.
    """
    n_chunks = len(chunks)
    normalizer = (1.0 / n_chunks) if normalize_episode_length else 1.0
    chunk_scale = grad_coef * normalizer
    for chunk in chunks:
        mean = policy(chunk["input"])
        chunk_lp = gaussian_logprob(
            chunk["action"], mean, std,
            mask=chunk["mask"],
            normalize_action_dim=normalize_action_dim,
        ).reshape(())
        (chunk_lp * chunk_scale).backward()


def _make_synthetic_episode(policy, *, n_chunks, in_dim, std, seed):
    """Sample chunk inputs and record actions as Normal(policy(input), std)."""
    g = torch.Generator().manual_seed(seed)
    chunks = []
    with torch.no_grad():
        for _ in range(n_chunks):
            x = torch.randn(in_dim, generator=g)
            mu_old = policy(x)
            noise = torch.randn(mu_old.shape, generator=g) * std
            action = mu_old + noise
            mask = torch.ones_like(mu_old, dtype=torch.bool)
            chunks.append({"input": x, "action": action.detach(), "mask": mask})
    return chunks


def test_grpo_per_chunk_backward_increases_logprob_for_positive_adv():
    """Smoking-gun: positive advantage MUST raise trajectory logprob after one step.

    Mirrors _backward_episode_logprob exactly (per-chunk forward + backward of
    ``chunk_lp * grad_coef / n_chunks``, then a single optimizer step). With
    ``grad_coef = -1`` (the encoding of +1 advantage in the server, since
    ``grad_coef = -ratio * adv``), the policy parameters move in the gradient-
    ascent direction for chunk_lp, so re-evaluating at the same recorded
    actions MUST yield higher logprob — independent of mbs ordering, bf16
    drift, or KL-decay artifacts. If this assertion fails, there is a sign or
    normalization bug in the GRPO update path itself.
    """
    torch.manual_seed(0)
    std = 0.05
    n_chunks = 3
    in_dim = 16
    out_shape = (1, 30, 2, 16, 1)

    policy = _SyntheticChunkPolicy(in_dim, out_shape)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=8e-5)

    chunks = _make_synthetic_episode(
        policy, n_chunks=n_chunks, in_dim=in_dim, std=std, seed=42,
    )

    with torch.no_grad():
        old_lp = float(_episode_logprob_for_test(
            policy, chunks, std,
            normalize_action_dim=True, normalize_episode_length=True,
        ))

    optimizer.zero_grad(set_to_none=True)
    _backward_episode_for_test(
        policy, chunks, torch.tensor(-1.0), std,
        normalize_action_dim=True, normalize_episode_length=True,
    )
    optimizer.step()

    with torch.no_grad():
        new_lp = float(_episode_logprob_for_test(
            policy, chunks, std,
            normalize_action_dim=True, normalize_episode_length=True,
        ))

    log_ratio = new_lp - old_lp
    assert log_ratio > 0, (
        f"Positive advantage (grad_coef=-1) MUST raise logprob; "
        f"got log_ratio={log_ratio:.6e} (old={old_lp:.6f}, new={new_lp:.6f})"
    )


def test_grpo_per_chunk_backward_decreases_logprob_for_negative_adv():
    """Mirror of the positive-adv test: grad_coef=+1 must lower logprob."""
    torch.manual_seed(1)
    std = 0.05
    n_chunks = 3
    in_dim = 16
    out_shape = (1, 30, 2, 16, 1)

    policy = _SyntheticChunkPolicy(in_dim, out_shape)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=8e-5)

    chunks = _make_synthetic_episode(
        policy, n_chunks=n_chunks, in_dim=in_dim, std=std, seed=43,
    )

    with torch.no_grad():
        old_lp = float(_episode_logprob_for_test(
            policy, chunks, std,
            normalize_action_dim=True, normalize_episode_length=True,
        ))

    optimizer.zero_grad(set_to_none=True)
    _backward_episode_for_test(
        policy, chunks, torch.tensor(+1.0), std,
        normalize_action_dim=True, normalize_episode_length=True,
    )
    optimizer.step()

    with torch.no_grad():
        new_lp = float(_episode_logprob_for_test(
            policy, chunks, std,
            normalize_action_dim=True, normalize_episode_length=True,
        ))

    log_ratio = new_lp - old_lp
    assert log_ratio < 0, (
        f"Negative advantage (grad_coef=+1) MUST lower logprob; "
        f"got log_ratio={log_ratio:.6e} (old={old_lp:.6f}, new={new_lp:.6f})"
    )
