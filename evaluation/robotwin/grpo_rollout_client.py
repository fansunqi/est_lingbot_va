"""RoboTwin rollout client for online LingBot-VA GRPO."""

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

if os.environ.get("CONDA_DEFAULT_ENV") != "RoboTwin" and not os.environ.get("ALLOW_NON_ROBOTWIN_CONDA"):
    raise SystemExit(
        "GRPO rollout client must run in the conda RoboTwin environment. "
        "Run `conda activate RoboTwin` first, or set ALLOW_NON_ROBOTWIN_CONDA=1 "
        "if your RoboTwin environment has another name."
    )

robowin_root = Path("/home/cxy/WAM/RoboTwin")
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
) -> dict:
    TASK_ENV.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    TASK_ENV.set_instruction(instruction=instruction)
    prompt = TASK_ENV.get_instruction()
    model.reset_episode(
        prompt=prompt,
        task=task_name,
        seed=seed,
        group_id=group_id,
        episode_idx=episode_idx,
        rollout_seed=seed * 100000 + episode_idx,
    )

    first = True
    full_action_history = []
    full_obs_list = []
    initial_obs = TASK_ENV.get_obs()
    initial_formatted_obs = format_obs(initial_obs, prompt)
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
                full_action_history.append(raw_action_step)
                TASK_ENV.take_action(ee_action, action_type="ee")
                if (j + 1) % action_per_frame == 0:
                    obs = format_obs(TASK_ENV.get_obs(), prompt)
                    full_obs_list.append(obs)
                    key_frame_list.append(obs)

        first = False
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
    )

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
        "finish_response": finish_response,
        "video_path": str(video_path),
    }


def choose_instruction(task_name: str, episode_info, instruction_type: str = "seen") -> str:
    if isinstance(episode_info, dict) and "info" in episode_info:
        episode_info = episode_info["info"]
    results = generate_episode_descriptions(task_name, [episode_info], 1)
    return str(np.random.choice(results[0][instruction_type]))


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_render_check", action="store_true",
                        help="Skip Sapien_TEST ray-tracing probe; useful on headless nodes where task env rendering still works")
    args = parser.parse_args()

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
    model = GRPOWebsocketClientPolicy(host=args.host, port=args.port)

    global_episode = 0
    try:
        for item_idx, item in enumerate(assignment):
            task_name = item["task"]
            seed = int(item["seed"])
            episode_info = item.get("episode_info")
            if episode_info is None:
                raise RuntimeError("GRPO rollout assignment requires episode_info for same-seed replay")

            task_args = build_task_args(task_name, args.task_config)
            task_args["policy_name"] = args.policy_name
            task_args["save_root"] = args.save_root
            instruction = choose_instruction(task_name, episode_info)
            group_id = f"{task_name}:{seed}:{item_idx}:{instruction}"

            for group_member in range(args.group_size):
                TASK_ENV = None
                global_episode += 1
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
                        episode_idx=global_episode,
                        group_member=group_member,
                        save_root=str(run_dir),
                    )
                    succ = bool(episode_result["success"])
                    results[task_name]["succ"] += int(succ)
                    results[task_name]["total"] += 1
                    results[task_name]["episodes"].append({
                        "seed": seed,
                        "group_id": group_id,
                        "group_member": group_member,
                        "success": bool(succ),
                        "step_count": episode_result["step_count"],
                        "video_path": episode_result["video_path"],
                        "server_status": episode_result["finish_response"],
                    })
                except Exception as exc:
                    traceback.print_exc()
                    results[task_name]["total"] += 1
                    results[task_name]["episodes"].append({
                        "seed": seed,
                        "group_id": group_id,
                        "group_member": group_member,
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
    finally:
        model.close()


if __name__ == "__main__":
    main()
