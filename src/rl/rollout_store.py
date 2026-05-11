"""CPU rollout storage for same-task/same-seed GRPO groups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import torch


def _cpu_detach(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_detach(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_detach(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_detach(v) for v in value)
    return value


@dataclass
class RolloutChunk:
    obs: dict[str, Any]
    frame_st_id: int
    latent_noise: torch.Tensor
    action_chain: list[torch.Tensor]
    old_logprobs: torch.Tensor
    action_timesteps: torch.Tensor
    action_mask: torch.Tensor
    env_action: Any
    keyframes: list[dict[str, Any]] | None = None
    state: Any | None = None


@dataclass
class EpisodeRecord:
    episode_id: str
    session_id: str
    prompt: str
    task: str
    seed: int
    group_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list[RolloutChunk] = field(default_factory=list)
    success: bool | None = None
    reward: float | None = None
    step_count: int | None = None

    @property
    def complete(self) -> bool:
        return self.reward is not None


class RolloutStore:
    def __init__(self, group_size: int):
        self.group_size = int(group_size)
        self._active_by_session: dict[str, EpisodeRecord] = {}
        self._episodes: dict[str, EpisodeRecord] = {}
        self._completed_by_group: dict[str, list[str]] = {}

    def start_episode(
        self,
        *,
        session_id: str,
        prompt: str,
        task: str,
        seed: int,
        group_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeRecord:
        group_id = group_id or f"{task}:{seed}:{prompt}"
        episode = EpisodeRecord(
            episode_id=str(uuid4()),
            session_id=session_id,
            prompt=prompt,
            task=task,
            seed=int(seed),
            group_id=group_id,
            metadata=metadata or {},
        )
        self._active_by_session[session_id] = episode
        self._episodes[episode.episode_id] = episode
        return episode

    def active(self, session_id: str) -> EpisodeRecord | None:
        return self._active_by_session.get(session_id)

    def add_chunk(self, session_id: str, chunk: RolloutChunk) -> None:
        episode = self.active(session_id)
        if episode is None:
            raise RuntimeError(f"No active GRPO episode for session {session_id!r}")
        episode.chunks.append(_cpu_detach(chunk))

    def attach_chunk_context(self, session_id: str, *, keyframes, state) -> None:
        episode = self.active(session_id)
        if episode is None or not episode.chunks:
            return
        episode.chunks[-1].keyframes = _cpu_detach(keyframes)
        episode.chunks[-1].state = _cpu_detach(state)

    def finish_episode(
        self,
        session_id: str,
        *,
        success: bool,
        step_count: int | None = None,
        reward: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[EpisodeRecord, list[EpisodeRecord] | None]:
        episode = self.active(session_id)
        if episode is None:
            raise RuntimeError(f"No active GRPO episode for session {session_id!r}")
        episode.success = bool(success)
        episode.reward = float(success) if reward is None else float(reward)
        episode.step_count = step_count
        if metadata:
            episode.metadata.update(metadata)
        self._active_by_session.pop(session_id, None)
        group = self._completed_by_group.setdefault(episode.group_id, [])
        group.append(episode.episode_id)
        if len(group) >= self.group_size:
            ready_ids = group[: self.group_size]
            del group[: self.group_size]
            return episode, [self._episodes[eid] for eid in ready_ids]
        return episode, None

    def remove_session(self, session_id: str) -> None:
        self._active_by_session.pop(session_id, None)

    def state_dict(self) -> dict[str, Any]:
        return {
            "group_size": self.group_size,
            "episodes": self._episodes,
            "completed_by_group": self._completed_by_group,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.group_size = int(state["group_size"])
        self._episodes = state.get("episodes", {})
        self._completed_by_group = state.get("completed_by_group", {})
        self._active_by_session = {}
