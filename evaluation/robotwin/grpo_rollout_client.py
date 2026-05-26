"""RoboTwin rollout client for online LingBot-VA GRPO."""

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

if os.environ.get("CONDA_DEFAULT_ENV") != "RoboTwin" and not os.environ.get("ALLOW_NON_ROBOTWIN_CONDA"):
    raise SystemExit(
        "GRPO rollout client must run in the conda RoboTwin environment. "
        "Run `conda activate RoboTwin` first, or set ALLOW_NON_ROBOTWIN_CONDA=1 "
        "if your RoboTwin environment has another name."
    )

robowin_root = Path("/root/WAM/RoboTwin")
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))
os.chdir(robowin_root)

import numpy as np
import torch
from description.utils.generate_episode_instructions import *

from evaluation.robotwin.eval_session_client import (
    add_init_pose,
    build_task_args,
    class_decorator,
    cleanup_env_resources,
    format_obs,
    write_json,
)
from evaluation.robotwin.geometry import euler2quat
from evaluation.robotwin.grpo_websocket_client_policy import GRPOWebsocketClientPolicy
from evaluation.robotwin.test_render import Sapien_TEST


class _RoboTwinStepFilter:
    """Suppress per-env-step progress writes while preserving normal stdout."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, text):
        stripped = text.lstrip()
        if stripped.startswith("step:") or text == "\r":
            return len(text)
        return self._wrapped.write(text)

    def flush(self):
        return self._wrapped.flush()

    def isatty(self):
        return self._wrapped.isatty()

    @property
    def encoding(self):
        return getattr(self._wrapped, "encoding", None)


def _env_action_from_model_action(action, i, j, init_eef_pose):
    raw_action_step = action[:, i, j].flatten()
    ee_action = action[:, i, j]
    if action.shape[0] == 14:
        ee_action = np.concatenate([
            ee_action[:3],
            euler2quat(ee_action[3], ee_action[4], ee_action[5]),
            ee_action[6:10],
            euler2quat(ee_action[10], ee_action[11], ee_action[12]),
            ee_action[13:14],
        ])
    elif action.shape[0] == 16:
        ee_action = add_init_pose(ee_action, init_eef_pose)
        ee_action = np.concatenate([
            ee_action[:3],
            ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
            ee_action[7:11],
            ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
            ee_action[15:16],
        ])
    else:
        raise NotImplementedError(f"Unsupported action channel count: {action.shape[0]}")
    return raw_action_step, ee_action


def run_one_grpo_episode(
    TASK_ENV,
    model: GRPOWebsocketClientPolicy,
    *,
    task_name: str,
    seed: int,
    args: dict,
    episode_info,
    instruction: str,
    group_id: str,
    episode_idx: int,
    group_member: int,
    save_root: str,
    visualization_mode: str,
    max_episode_steps: int | None = None,
    is_eval: bool = False,
) -> dict:
    episode_start = time.monotonic()
    TASK_ENV.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    if max_episode_steps is not None and max_episode_steps > 0:
        # Override RoboTwin's per-task step_lim (task_config/_eval_step_limit.yml).
        # setup_demo() reads that file when is_test=True, so the override must
        # come after setup_demo.
        TASK_ENV.step_lim = int(max_episode_steps)
    TASK_ENV.set_instruction(instruction=instruction)
    prompt = TASK_ENV.get_instruction()
    model.reset_episode(
        prompt=prompt,
        task=task_name,
        seed=seed,
        group_id=group_id,
        episode_idx=episode_idx,
        rollout_seed=seed * 100000 + episode_idx,
        is_eval=is_eval,
    )

    first = True
    collect_visualization = visualization_mode.lower() not in ("0", "false", "off", "none", "no")
    full_action_history = [] if collect_visualization else None
    full_obs_list = [] if collect_visualization else None
    initial_obs = TASK_ENV.get_obs()
    initial_formatted_obs = format_obs(initial_obs, prompt)
    if full_obs_list is not None:
        full_obs_list.append(initial_formatted_obs)
    init_eef_pose = (
        initial_obs["endpose"]["left_endpose"]
        + [initial_obs["endpose"]["left_gripper"]]
        + initial_obs["endpose"]["right_endpose"]
        + [initial_obs["endpose"]["right_gripper"]]
    )
    init_eef_pose = np.array(init_eef_pose, dtype=np.float64)
    first_obs = None
    succ = False
    chunk_count = 0

    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        if first:
            observation = TASK_ENV.get_obs()
            first_obs = format_obs(observation, prompt)

        ret = model.sample_action({"obs": first_obs}, prompt=prompt)
        action = ret["action"]
        key_frame_list = []

        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                raw_action_step, ee_action = _env_action_from_model_action(action, i, j, init_eef_pose)
                if full_action_history is not None:
                    full_action_history.append(raw_action_step)
                TASK_ENV.take_action(ee_action, action_type="ee")
                if (j + 1) % action_per_frame == 0:
                    obs = format_obs(TASK_ENV.get_obs(), prompt)
                    if full_obs_list is not None:
                        full_obs_list.append(obs)
                    key_frame_list.append(obs)

        first = False
        chunk_count += 1
        model.commit_chunk(obs=key_frame_list, state=action)

        if TASK_ENV.eval_success:
            succ = True
            break

    step_count = int(TASK_ENV.take_action_cnt)
    finish_response = model.finish_episode(
        success=succ,
        step_count=step_count,
        task=task_name,
        seed=seed,
        instruction=prompt,
        group_id=group_id,
        episode_idx=episode_idx,
        group_member=group_member,
    )

    video_path = None
    if collect_visualization and should_save_visualization(
        visualization_mode,
        success=succ,
        episode_idx=episode_idx,
    ):
        vis_dir = Path(save_root) / "visualization" / task_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        video_name = f"{episode_idx}_member{group_member}_{prompt.replace(' ', '_')}_{succ}.mp4"
        video_path = vis_dir / video_name

        from evaluation.robotwin.eval_polict_client_openpi import save_comparison_video
        save_comparison_video(
            real_obs_list=full_obs_list,
            imagined_video=None,
            action_history=full_action_history,
            save_path=str(video_path),
            fps=15,
        )

    cleanup_env_resources(TASK_ENV, clear_cache=True)
    del full_action_history, full_obs_list, first_obs, initial_obs, initial_formatted_obs
    return {
        "success": bool(succ),
        "step_count": step_count,
        "chunk_count": chunk_count,
        "elapsed_s": time.monotonic() - episode_start,
        "finish_response": finish_response,
        "video_path": str(video_path) if video_path is not None else None,
    }


def should_save_visualization(mode: str, *, success: bool, episode_idx: int) -> bool:
    mode = mode.lower()
    if mode in ("0", "false", "off", "none", "no"):
        return False
    if mode in ("1", "true", "on", "all", "yes"):
        return True
    if mode in ("success", "successes"):
        return bool(success)
    if mode in ("failure", "failures"):
        return not bool(success)
    if mode.startswith("every_"):
        interval = int(mode.removeprefix("every_"))
        return interval > 0 and episode_idx % interval == 0
    raise ValueError(
        "--save_visualization must be one of none/all/success/failure/every_N, "
        f"got {mode!r}"
    )


def choose_instruction(
    task_name: str,
    episode_info,
    instruction_type: str = "seen",
    rng=None,
    seed: int | None = None,
) -> str:
    if isinstance(episode_info, dict) and "info" in episode_info:
        episode_info = episode_info["info"]
    if seed is None:
        results = generate_episode_descriptions(task_name, [episode_info], 1)
    else:
        # RoboTwin's generator uses the global Python random module internally
        # (shuffle + object-description choice), so make that deterministic
        # across parallel rollout clients without leaking state afterward.
        random_state = random.getstate()
        try:
            random.seed(int(seed))
            results = generate_episode_descriptions(task_name, [episode_info], 1)
        finally:
            random.setstate(random_state)
    choices = results[0][instruction_type]
    if rng is None:
        return str(np.random.choice(choices))
    return str(rng.choice(choices))


def wait_for_group_barrier(
    run_dir: Path,
    *,
    item_idx: int,
    client_id: int,
    num_clients: int,
    timeout_s: float = 0.0,
) -> None:
    if num_clients <= 1:
        return
    barrier_dir = run_dir / "barriers" / f"group_{item_idx:06d}"
    barrier_dir.mkdir(parents=True, exist_ok=True)
    marker = barrier_dir / f"client_{client_id}.done"
    marker.write_text(f"{time.time()}\n")

    start = time.monotonic()
    last_notice = -1
    while True:
        done = list(barrier_dir.glob("client_*.done"))
        if len(done) >= num_clients:
            print(
                f"Group barrier passed for item {item_idx}: "
                f"{len(done)}/{num_clients} clients ready."
            )
            return
        if timeout_s > 0 and time.monotonic() - start > timeout_s:
            raise RuntimeError(
                f"Timed out waiting at group barrier item {item_idx}: "
                f"{len(done)}/{num_clients} clients ready in {barrier_dir}"
            )
        elapsed_min = int((time.monotonic() - start) // 60)
        if elapsed_min != last_notice:
            last_notice = elapsed_min
            print(
                f"Waiting at group barrier item {item_idx}: "
                f"{len(done)}/{num_clients} clients ready."
            )
        time.sleep(5)


def run_update_after_group_barrier(
    model: GRPOWebsocketClientPolicy,
    run_dir: Path,
    *,
    item_idx: int,
    client_id: int,
    num_clients: int,
    timeout_s: float = 0.0,
) -> dict:
    """Run the GRPO update on client 0; peers wait then read the same status.

    Returns the status dict from `run_pending_updates`. Non-zero clients
    deserialize update_status.json so they can also see fields like
    `in_eval` and `global_update_step` — needed to coordinate parallel eval.
    """
    if num_clients <= 1:
        status = model.run_pending_updates()
        print(f"GRPO update trigger after item {item_idx}: {status}")
        return status

    barrier_dir = run_dir / "barriers" / f"group_{item_idx:06d}"
    update_done = barrier_dir / "update.done"
    update_status = barrier_dir / "update_status.json"
    update_error = barrier_dir / "update_error.txt"

    if client_id == 0:
        try:
            status = model.run_pending_updates()
            write_json(status, update_status)
            print(f"GRPO update trigger after item {item_idx}: {status}")
        except Exception:
            update_error.write_text(traceback.format_exc())
            raise
        finally:
            update_done.write_text(f"{time.time()}\n")
        return status

    start = time.monotonic()
    last_notice = -1
    while True:
        if update_done.exists():
            if update_error.exists():
                raise RuntimeError(update_error.read_text())
            print(f"GRPO update barrier passed for item {item_idx}.")
            with open(update_status, "r") as f:
                return json.load(f)
        if timeout_s > 0 and time.monotonic() - start > timeout_s:
            raise RuntimeError(
                f"Timed out waiting for GRPO update after item {item_idx}: {barrier_dir}"
            )
        elapsed_min = int((time.monotonic() - start) // 60)
        if elapsed_min != last_notice:
            last_notice = elapsed_min
            print(f"Waiting for GRPO update after item {item_idx}.")
        time.sleep(5)


def run_eval_pass(
    model: GRPOWebsocketClientPolicy,
    assignment_with_idx: list[tuple[int, dict]],
    args,
    *,
    run_dir: Path,
    global_update_step: int,
) -> dict:
    """One deterministic pass over the (item_idx, item) pairs assigned to this client.

    `item_idx_in_pass` is the index into the *full* assignment, not the position
    within the slice — this keeps `instruction_seed`, `eval_group_id`, and
    `episode_idx` deterministic across slicing modes (single-client uses the
    full assignment; multi-client uses `assignment[client_id::num_clients]`).

    Server-side, every `reset_episode(is_eval=True)` flips the session into the
    no-noise sampling path and routes the finished episode into
    `_eval_results` instead of `_pending_ready_groups`. `end_eval_phase` is
    called by client 0 only, after every client signals its slice is done.
    """
    pass_start = time.monotonic()
    print(
        f"[eval] starting deterministic pass: global_update_step={global_update_step} "
        f"items={len(assignment_with_idx)} client_id={args.client_id}"
    )
    metrics_dir = run_dir / "metrics" / "eval"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"step_{global_update_step:06d}_client_{args.client_id}.json"

    per_episode: list[dict] = []
    successes = 0
    total = 0
    for item_idx_in_pass, item in assignment_with_idx:
        task_name = item["task"]
        seed = int(item["seed"])
        episode_info = item.get("episode_info")
        if episode_info is None:
            raise RuntimeError("GRPO eval pass requires episode_info on every assignment item")

        task_args = build_task_args(task_name, args.task_config)
        task_args["policy_name"] = args.policy_name
        task_args["save_root"] = args.save_root
        # Mirror training's instruction_seed scheme so eval and training use
        # the same instruction for the same item_idx (comparable across passes).
        instruction_seed = int(args.seed) * 1000003 + item_idx_in_pass
        instruction_rng = np.random.default_rng(instruction_seed)
        instruction = item.get("instruction")
        if instruction is None:
            instruction = choose_instruction(
                task_name,
                episode_info,
                rng=instruction_rng,
                seed=instruction_seed,
            )
        instruction = str(instruction)
        # Distinct group_id keeps eval episodes from accidentally colliding
        # with training group_ids on the server (eval ones are dropped
        # immediately, but the name is what gets logged).
        eval_group_id = f"eval:{global_update_step}:{task_name}:{seed}:{item_idx_in_pass}:{instruction}"
        episode_idx = item_idx_in_pass * args.group_size + 1  # mirror group_member=0 in training

        TASK_ENV = None
        try:
            TASK_ENV = class_decorator(task_name)
            episode_result = run_one_grpo_episode(
                TASK_ENV,
                model,
                task_name=task_name,
                seed=seed,
                args=task_args,
                episode_info=episode_info,
                instruction=instruction,
                group_id=eval_group_id,
                episode_idx=episode_idx,
                group_member=0,
                save_root=str(run_dir),
                visualization_mode="none",
                max_episode_steps=args.max_episode_steps,
                is_eval=True,
            )
            succ = bool(episode_result["success"])
            successes += int(succ)
            total += 1
            per_episode.append({
                "item_idx": item_idx_in_pass,
                "task": task_name,
                "seed": seed,
                "instruction": instruction,
                "success": succ,
                "step_count": episode_result["step_count"],
                "elapsed_s": episode_result["elapsed_s"],
            })
            print(
                f"[eval] item={item_idx_in_pass} task={task_name} seed={seed} "
                f"success={succ} steps={episode_result['step_count']} "
                f"elapsed_s={episode_result['elapsed_s']:.1f}"
            )
        except Exception as exc:
            traceback.print_exc()
            print(f"[eval] item={item_idx_in_pass} task={task_name} seed={seed} error={exc}")
            total += 1
            per_episode.append({
                "item_idx": item_idx_in_pass,
                "task": task_name,
                "seed": seed,
                "success": False,
                "error": str(exc),
            })
        finally:
            if TASK_ENV is not None:
                cleanup_env_resources(TASK_ENV, clear_cache=True)
                del TASK_ENV
            torch.cuda.empty_cache()
        write_json(
            {
                "global_update_step": global_update_step,
                "client_id": args.client_id,
                "total": total,
                "successes": successes,
                "success_rate": (successes / total) if total else 0.0,
                "episodes": per_episode,
            },
            metrics_path,
        )

    pass_elapsed = time.monotonic() - pass_start
    print(
        f"[eval] pass done: global_update_step={global_update_step} "
        f"total={total} successes={successes} "
        f"success_rate={successes/total if total else 0.0:.4f} "
        f"elapsed_s={pass_elapsed:.1f}"
    )
    return {
        "global_update_step": global_update_step,
        "total": total,
        "successes": successes,
        "success_rate": (successes / total) if total else 0.0,
        "elapsed_s": pass_elapsed,
    }


def _wait_for_eval_slice_dones(
    barrier_dir: Path, *, num_clients: int, timeout_s: float = 0.0
) -> None:
    """Block until every client has written client_<id>.done into barrier_dir."""
    if num_clients <= 1:
        return
    start = time.monotonic()
    last_notice = -1
    while True:
        done = list(barrier_dir.glob("client_*.done"))
        if len(done) >= num_clients:
            print(
                f"[eval] all clients finished slice: {len(done)}/{num_clients} "
                f"ready in {barrier_dir.name}"
            )
            return
        if timeout_s > 0 and time.monotonic() - start > timeout_s:
            raise RuntimeError(
                f"Timed out waiting for eval slice completion: "
                f"{len(done)}/{num_clients} ready in {barrier_dir}"
            )
        elapsed_min = int((time.monotonic() - start) // 60)
        if elapsed_min != last_notice:
            last_notice = elapsed_min
            print(
                f"[eval] waiting for clients to finish slice: "
                f"{len(done)}/{num_clients} done in {barrier_dir.name}"
            )
        time.sleep(5)


def _wait_for_eval_end(barrier_dir: Path, *, timeout_s: float = 0.0) -> None:
    """Block until client 0 has written end.done (i.e. end_eval_phase returned)."""
    end_done = barrier_dir / "end.done"
    start = time.monotonic()
    last_notice = -1
    while True:
        if end_done.exists():
            return
        if timeout_s > 0 and time.monotonic() - start > timeout_s:
            raise RuntimeError(f"Timed out waiting for eval end barrier: {end_done}")
        elapsed_min = int((time.monotonic() - start) // 60)
        if elapsed_min != last_notice:
            last_notice = elapsed_min
            print(
                f"[eval] waiting for client 0 to finish end_eval_phase: {end_done}"
            )
        time.sleep(5)


def wait_and_run_eval_pass(
    model: GRPOWebsocketClientPolicy,
    assignment: list,
    args,
    *,
    run_dir: Path,
    global_update_step: int,
    timeout_s: float = 0.0,
) -> dict | None:
    """Multi-client eval with file barrier.

    Each client runs `assignment[client_id::num_clients]` with is_eval=True so
    the work is parallel. Client 0 calls `end_eval_phase` only after every
    client signals its slice is done — otherwise the server would flush
    `eval/*` to wandb while peers were still sending eval episodes. Non-zero
    clients block on `end.done` so they don't resume RL rollouts (and hand
    fresh non-eval episodes to `_pending_ready_groups`) before the eval phase
    is officially closed on the server.

    For `num_clients == 1` this collapses to the original single-client path
    (slice == full assignment, barrier is a no-op, end_eval_phase fires
    immediately).
    """
    barrier_dir = run_dir / "barriers" / f"eval_{global_update_step:06d}"
    barrier_dir.mkdir(parents=True, exist_ok=True)

    full_indexed = list(enumerate(assignment))
    slice_with_idx = full_indexed[args.client_id :: args.num_clients]
    print(
        f"[eval] client {args.client_id}/{args.num_clients} taking "
        f"{len(slice_with_idx)}/{len(assignment)} items "
        f"at global_update_step={global_update_step}"
    )

    try:
        run_eval_pass(
            model,
            slice_with_idx,
            args,
            run_dir=run_dir,
            global_update_step=global_update_step,
        )
    finally:
        # Mark our slice done even on failure so peers don't block forever
        # on a crashed client. The crashed client will raise after this.
        (barrier_dir / f"client_{args.client_id}.done").write_text(f"{time.time()}\n")

    if args.client_id == 0:
        _wait_for_eval_slice_dones(
            barrier_dir, num_clients=args.num_clients, timeout_s=timeout_s
        )
        end_status = model.end_eval_phase()
        write_json(end_status, barrier_dir / "end_status.json")
        (barrier_dir / "end.done").write_text(f"{time.time()}\n")
        print(f"[eval] phase done: {end_status}")
        return end_status
    else:
        _wait_for_eval_end(barrier_dir, timeout_s=timeout_s)
        return None


def main():
    parser = argparse.ArgumentParser(description="RoboTwin GRPO rollout client")
    parser.add_argument("--assignment", type=str, required=True)
    parser.add_argument("--port", type=int, default=29546)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--save_root", type=str, default="./results/grpo")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument("--policy_name", type=str, default="ACT")
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--num_clients", type=int, default=1,
                        help="Total number of parallel clients sharing each GRPO group")
    parser.add_argument("--group_barrier", action="store_true",
                        help="Wait for all parallel clients after each assignment item")
    parser.add_argument("--group_barrier_timeout", type=float, default=0.0,
                        help="Seconds before failing the group barrier; 0 means wait forever")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_passes",
        type=int,
        default=1,
        help="How many times to loop over the entire assignment. Each pass "
        "uses a globally incrementing item_idx so group_id stays unique, i.e. "
        "the server treats pass 2's (seed=20000) as a fresh group rather than "
        "merging it with pass 1's.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Run one deterministic GRPO eval pass and exit without training rollouts.",
    )
    parser.add_argument("--skip_render_check", action="store_true",
                        help="Skip Sapien_TEST ray-tracing probe; useful on headless nodes where task env rendering still works")
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=None,
        help="Override RoboTwin's per-task step_lim (task_config/_eval_step_limit.yml). "
        "Episodes are forced to terminate after this many env steps without modifying "
        "the upstream RoboTwin config. Default: use RoboTwin's value.",
    )
    parser.add_argument(
        "--save_visualization",
        type=str,
        default=os.environ.get("GRPO_SAVE_VISUALIZATION", "none"),
        help="Rollout video policy: none, all, success, failure, or every_N. Default: none",
    )
    args = parser.parse_args()
    try:
        should_save_visualization(args.save_visualization, success=False, episode_idx=1)
    except ValueError as exc:
        parser.error(str(exc))
    if os.environ.get("GRPO_SHOW_ENV_STEPS") != "1":
        sys.stdout = _RoboTwinStepFilter(sys.stdout)

    if args.group_size < 1:
        raise ValueError("--group_size must be >= 1")
    if args.num_clients < 1:
        raise ValueError("--num_clients must be >= 1")
    if not (0 <= args.client_id < args.num_clients):
        raise ValueError(f"--client_id must be in [0, {args.num_clients}), got {args.client_id}")
    if args.num_passes < 1:
        raise ValueError("--num_passes must be >= 1")

    with open(args.assignment, "r") as f:
        assignment = json.load(f)

    run_dir = Path(args.run_dir) if args.run_dir else Path(args.save_root)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"client_{args.client_id}.json"
    results = defaultdict(lambda: {"succ": 0, "total": 0, "episodes": []})

    if args.skip_render_check or os.environ.get("SKIP_RENDER_CHECK") == "1":
        print("Skipping Sapien_TEST render probe.")
    else:
        Sapien_TEST()

    group_members = list(range(args.client_id, args.group_size, args.num_clients))
    if not group_members:
        print(
            f"Client {args.client_id} has no group members for "
            f"group_size={args.group_size}, num_clients={args.num_clients}; exiting."
        )
        return
    print(
        f"Client {args.client_id}/{args.num_clients} will run group members "
        f"{group_members} for each assignment item."
    )

    model = GRPOWebsocketClientPolicy(host=args.host, port=args.port)

    try:
        # If the server armed an eval phase at startup (eval_every > 0 on a
        # fresh run), drain it before any RL rollouts so we have a baseline
        # success rate at global_update_step=0. Every client polls the server
        # independently: the server's _eval_pending flag stays True until
        # client 0 calls end_eval_phase, and wait_and_run_eval_pass's barrier
        # makes sure that only happens after every client has finished its
        # slice — so even a slow-starting client can never miss this.
        initial_eval = model.get_eval_phase()
        if initial_eval.get("in_eval"):
            gus = int(initial_eval.get("global_update_step", 0))
            eval_steps = initial_eval.get("eval_action_num_inference_steps")
            print(
                f"[eval] initial eval armed at global_update_step={gus}; "
                f"client {args.client_id}/{args.num_clients} joining "
                f"parallel deterministic pass over {len(assignment)} items "
                f"(action_steps={eval_steps})"
            )
            wait_and_run_eval_pass(
                model,
                assignment,
                args,
                run_dir=run_dir,
                global_update_step=gus,
                timeout_s=args.group_barrier_timeout,
            )
            if args.eval_only:
                print("[eval] eval_only requested; exiting after initial eval pass.")
                return
        elif args.eval_only:
            status = model.get_status()
            gus = int(status.get("global_update_step", 0))
            eval_steps = status.get("eval_action_num_inference_steps")
            print(
                f"[eval] eval_only requested; no pending eval phase, so running "
                f"one deterministic pass at global_update_step={gus} "
                f"(action_steps={eval_steps})"
            )
            wait_and_run_eval_pass(
                model,
                assignment,
                args,
                run_dir=run_dir,
                global_update_step=gus,
                timeout_s=args.group_barrier_timeout,
            )
            return

        items_per_pass = len(assignment)
        for pass_idx in range(args.num_passes):
            if args.num_passes > 1:
                print(
                    f"Client {args.client_id} starting pass {pass_idx + 1}/{args.num_passes} "
                    f"({items_per_pass} items per pass)."
                )
            for item_idx_in_pass, item in enumerate(assignment):
                # Globally-monotonic item_idx so group_id is unique across passes;
                # the server keys groups by (task, seed, item_idx, instruction).
                item_idx = pass_idx * items_per_pass + item_idx_in_pass
                task_name = item["task"]
                seed = int(item["seed"])
                episode_info = item.get("episode_info")
                if episode_info is None:
                    raise RuntimeError("GRPO rollout assignment requires episode_info for same-seed replay")

                task_args = build_task_args(task_name, args.task_config)
                task_args["policy_name"] = args.policy_name
                task_args["save_root"] = args.save_root
                instruction_seed = int(args.seed) * 1000003 + item_idx
                instruction_rng = np.random.default_rng(instruction_seed)
                instruction = item.get("instruction")
                if instruction is None:
                    instruction = choose_instruction(
                        task_name,
                        episode_info,
                        rng=instruction_rng,
                        seed=instruction_seed,
                    )
                instruction = str(instruction)
                group_id = f"{task_name}:{seed}:{item_idx}:{instruction}"
                print(
                    f"Client {args.client_id} group start: item={item_idx} task={task_name} seed={seed} "
                    f"group_id={group_id} members={group_members} prompt={instruction!r}"
                )

                for group_member in group_members:
                    TASK_ENV = None
                    episode_idx = item_idx * args.group_size + group_member + 1
                    try:
                        TASK_ENV = class_decorator(task_name)
                        episode_result = run_one_grpo_episode(
                            TASK_ENV,
                            model,
                            task_name=task_name,
                            seed=seed,
                            args=task_args,
                            episode_info=episode_info,
                            instruction=instruction,
                            group_id=group_id,
                            episode_idx=episode_idx,
                            group_member=group_member,
                            save_root=str(run_dir),
                            visualization_mode=args.save_visualization,
                            max_episode_steps=args.max_episode_steps,
                        )
                        succ = bool(episode_result["success"])
                        results[task_name]["succ"] += int(succ)
                        results[task_name]["total"] += 1
                        results[task_name]["episodes"].append({
                            "seed": seed,
                            "group_id": group_id,
                            "group_member": group_member,
                            "group_size": args.group_size,
                            "num_clients": args.num_clients,
                            "success": bool(succ),
                            "step_count": episode_result["step_count"],
                            "chunk_count": episode_result["chunk_count"],
                            "elapsed_s": episode_result["elapsed_s"],
                            "video_path": episode_result["video_path"],
                            "server_status": episode_result["finish_response"],
                        })
                        server_status = episode_result["finish_response"]
                        video_msg = episode_result["video_path"] or "disabled"
                        print(
                            f"Client {args.client_id} member done: item={item_idx} member={group_member} "
                            f"episode_idx={episode_idx} success={succ} steps={episode_result['step_count']} "
                            f"chunks={episode_result['chunk_count']} elapsed_s={episode_result['elapsed_s']:.1f} "
                            f"server_ready={server_status.get('ready_for_update')} "
                            f"pending_ready_groups={server_status.get('pending_ready_groups')} "
                            f"video={video_msg}"
                        )
                    except Exception as exc:
                        traceback.print_exc()
                        print(
                            f"Client {args.client_id} member failed: item={item_idx} member={group_member} "
                            f"episode_idx={episode_idx} error={exc}"
                        )
                        results[task_name]["total"] += 1
                        results[task_name]["episodes"].append({
                            "seed": seed,
                            "group_id": group_id,
                            "group_member": group_member,
                            "group_size": args.group_size,
                            "num_clients": args.num_clients,
                            "success": False,
                            "error": str(exc),
                        })
                        cleanup_env_resources(TASK_ENV, clear_cache=True)
                    finally:
                        if TASK_ENV is not None:
                            cleanup_env_resources(TASK_ENV, clear_cache=True)
                            del TASK_ENV
                        write_json(dict(results), metrics_path)
                        torch.cuda.empty_cache()
                if args.group_barrier:
                    wait_for_group_barrier(
                        run_dir,
                        item_idx=item_idx,
                        client_id=args.client_id,
                        num_clients=args.num_clients,
                        timeout_s=args.group_barrier_timeout,
                    )
                    update_status = run_update_after_group_barrier(
                        model,
                        run_dir,
                        item_idx=item_idx,
                        client_id=args.client_id,
                        num_clients=args.num_clients,
                        timeout_s=args.group_barrier_timeout,
                    )
                    # All clients see in_eval through the barrier file written
                    # by client 0 in run_update_after_group_barrier, so they
                    # all pause RL and join the eval pass in parallel.
                    if update_status.get("in_eval"):
                        gus = int(update_status.get("global_update_step", 0))
                        print(
                            f"[eval] phase armed at global_update_step={gus}; "
                            f"client {args.client_id}/{args.num_clients} joining "
                            f"parallel deterministic pass over {len(assignment)} items"
                        )
                        wait_and_run_eval_pass(
                            model,
                            assignment,
                            args,
                            run_dir=run_dir,
                            global_update_step=gus,
                            timeout_s=args.group_barrier_timeout,
                        )
                elif args.client_id == 0:
                    # No-barrier path: only safe with num_clients==1. Multi-client
                    # without --group_barrier has no synchronization point and is
                    # not supported for eval (peers would keep training).
                    status = model.run_pending_updates()
                    print(f"GRPO update trigger after item {item_idx}: {status}")
                    if status.get("in_eval"):
                        gus = int(status.get("global_update_step", 0))
                        print(
                            f"[eval] phase armed at global_update_step={gus}; "
                            f"running deterministic pass over {len(assignment)} items"
                        )
                        wait_and_run_eval_pass(
                            model,
                            assignment,
                            args,
                            run_dir=run_dir,
                            global_update_step=gus,
                            timeout_s=args.group_barrier_timeout,
                        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
