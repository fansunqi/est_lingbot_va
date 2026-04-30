"""
Session-level evaluation client for RoboTwin.

Unlike eval_polict_client_openpi.py (one client per task), this client
reads a work assignment JSON file containing multiple (task, seed) pairs
and executes them sequentially. This enables session-level load balancing.

Usage:
    python -m evaluation.robotwin.eval_session_client \
        --config policy/ACT/deploy_policy.yml \
        --assignment task_assignments/client_0.json \
        --port 29556 \
        --save_root ./results \
        --task_config demo_clean
"""

import sys
import os
import subprocess
from pathlib import Path

robowin_root = Path("/home/cxy/WAM/RoboTwin")
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))
os.chdir(robowin_root)

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
import torch
from collections import defaultdict
import traceback
import yaml
from datetime import datetime
import importlib
import argparse
import json

from evaluation.robotwin.geometry import euler2quat
from scipy.spatial.transform import Rotation as R
from description.utils.generate_episode_instructions import *

import imageio

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
from evaluation.robotwin.test_render import Sapien_TEST


def write_json(data: dict, fpath: Path) -> None:
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=2)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit(f"Failed to create env for task: {task_name}")
    return env_instance


def get_embodiment_file(embodiment_type, _embodiment_types):
    robot_file = _embodiment_types[embodiment_type]["file_path"]
    if robot_file is None:
        raise RuntimeError("No embodiment files")
    return robot_file


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def format_obs(observation, prompt):
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": observation["joint_action"]["vector"],
        "task": prompt,
    }


def add_eef_pose(new_pose, init_pose):
    new_pose_R = R.from_quat(new_pose[3:7][None])
    init_pose_R = R.from_quat(init_pose[3:7][None])
    out_rot = (init_pose_R * new_pose_R).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def add_init_pose(new_pose, init_pose):
    left_pose = add_eef_pose(new_pose[:8], init_pose[:8])
    right_pose = add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left_pose, right_pose])


def build_task_args(task_name, task_config):
    """Build the args dict for a given task (config loading)."""
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["eval_mode"] = True

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0], _embodiment_types)
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0], _embodiment_types)
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0], _embodiment_types)
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1], _embodiment_types)
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    return args


def run_one_episode(TASK_ENV, model, seed, args, save_root,
                    video_guidance_scale=5.0, action_guidance_scale=5.0,
                    save_visualization=True, episode_idx=0):
    """
    Run a single evaluation episode for a given (task, seed).

    Returns:
        bool: whether the episode was successful
    """
    task_name = args['task_name']
    render_freq = args.get("render_freq", 0)
    instruction_type = 'seen'

    # Setup the episode (no expert check - seeds are pre-validated)
    TASK_ENV.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)

    # Get instruction by running expert once for episode info
    episode_info = TASK_ENV.play_once()
    TASK_ENV.close_env()

    # Re-setup with render
    args["render_freq"] = render_freq
    TASK_ENV.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    episode_info_list = [episode_info["info"]]
    results = generate_episode_descriptions(task_name, episode_info_list, 1)
    instruction = np.random.choice(results[0][instruction_type])
    TASK_ENV.set_instruction(instruction=instruction)

    succ = False
    prompt = TASK_ENV.get_instruction()
    ret = model.infer(dict(reset=True, prompt=prompt, save_visualization=save_visualization))

    first = True
    full_obs_list = []
    gen_video_list = []
    full_action_history = []

    initial_obs = TASK_ENV.get_obs()
    inint_eef_pose = (initial_obs['endpose']['left_endpose'] +
                      [initial_obs['endpose']['left_gripper']] +
                      initial_obs['endpose']['right_endpose'] +
                      [initial_obs['endpose']['right_gripper']])
    inint_eef_pose = np.array(inint_eef_pose, dtype=np.float64)
    initial_formatted_obs = format_obs(initial_obs, prompt)
    full_obs_list.append(initial_formatted_obs)
    first_obs = None

    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        if first:
            observation = TASK_ENV.get_obs()
            first_obs = format_obs(observation, prompt)

        ret = model.infer(dict(
            obs=first_obs, prompt=prompt,
            save_visualization=save_visualization,
            video_guidance_scale=video_guidance_scale,
            action_guidance_scale=action_guidance_scale
        ))
        action = ret['action']
        if 'video' in ret:
            gen_video_list.append(ret['video'])
        key_frame_list = []

        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4

        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                raw_action_step = action[:, i, j].flatten()
                full_action_history.append(raw_action_step)

                ee_action = action[:, i, j]
                if action.shape[0] == 14:
                    ee_action = np.concatenate([
                        ee_action[:3],
                        euler2quat(ee_action[3], ee_action[4], ee_action[5]),
                        ee_action[6:10],
                        euler2quat(ee_action[10], ee_action[11], ee_action[12]),
                        ee_action[13:14]
                    ])
                elif action.shape[0] == 16:
                    ee_action = add_init_pose(ee_action, inint_eef_pose)
                    ee_action = np.concatenate([
                        ee_action[:3],
                        ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                        ee_action[7:11],
                        ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                        ee_action[15:16]
                    ])
                else:
                    raise NotImplementedError
                TASK_ENV.take_action(ee_action, action_type='ee')

                if (j + 1) % action_per_frame == 0:
                    obs = format_obs(TASK_ENV.get_obs(), prompt)
                    full_obs_list.append(obs)
                    key_frame_list.append(obs)

        first = False
        model.infer(dict(
            obs=key_frame_list, compute_kv_cache=True,
            imagine=False, save_visualization=save_visualization,
            state=action
        ))

        if TASK_ENV.eval_success:
            succ = True
            break

    # Save visualization video
    vis_dir = Path(save_root) / 'visualization' / task_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    video_name = f"{episode_idx}_{prompt.replace(' ', '_')}_{succ}.mp4"
    out_img_file = vis_dir / video_name

    from evaluation.robotwin.eval_polict_client_openpi import save_comparison_video
    save_comparison_video(
        real_obs_list=full_obs_list,
        imagined_video=None,
        action_history=full_action_history,
        save_path=str(out_img_file),
        fps=15
    )

    TASK_ENV.close_env()

    if TASK_ENV.render_freq:
        TASK_ENV.viewer.close()

    if succ:
        print(f"\033[92m[{task_name} seed={seed}] Success!\033[0m")
    else:
        print(f"\033[91m[{task_name} seed={seed}] Fail!\033[0m")

    return succ


def main():
    parser = argparse.ArgumentParser(description="Session-level eval client")
    parser.add_argument("--config", type=str, required=True,
                        help="Policy config YAML path")
    parser.add_argument("--assignment", type=str, required=True,
                        help="Path to client assignment JSON file")
    parser.add_argument("--port", type=int, default=29556,
                        help="Server websocket port")
    parser.add_argument("--save_root", type=str, default="./results",
                        help="Root directory for saving results")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed multiplier (for stseed folder naming)")
    parser.add_argument("--client_id", type=int, default=0,
                        help="Client ID (for per-client metrics file naming)")
    parser.add_argument("--task_config", type=str, default="demo_clean",
                        help="Task config name")
    parser.add_argument("--video_guidance_scale", type=float, default=5.0)
    parser.add_argument("--action_guidance_scale", type=float, default=5.0)
    parser.add_argument("--policy_name", type=str, default="ACT")
    args = parser.parse_args()

    # Initialize renderer
    Sapien_TEST()

    # Load assignment
    with open(args.assignment, "r") as f:
        assignment = json.load(f)

    print(f"\033[32mLoaded assignment: {len(assignment)} episodes\033[0m")

    # Group assignment by task (they should already be sorted by task from balance_sessions)
    from itertools import groupby
    task_groups = []
    for task_name, group in groupby(assignment, key=lambda x: x["task"]):
        task_groups.append((task_name, list(group)))

    print(f"\033[32mTasks involved ({len(task_groups)}): "
          f"{[f'{name}({len(eps)})' for name, eps in task_groups]}\033[0m")

    # Connect to server
    model = WebsocketClientPolicy(port=args.port)
    print(f"\033[32mConnected to server on port {args.port}\033[0m")

    # All outputs go under save_root/stseed-{st_seed}/
    st_seed = 10000 * (1 + args.seed)
    run_dir = Path(args.save_root) / f"stseed-{st_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Execute assignment task-by-task: load env → run all episodes → unload env
    results = defaultdict(lambda: {"succ": 0, "total": 0, "episodes": []})
    global_idx = 0
    total_episodes = sum(len(eps) for _, eps in task_groups)

    for task_name, episodes in task_groups:
        print(f"\n\033[34m{'='*60}\033[0m")
        print(f"\033[34mLoading task environment: {task_name} ({len(episodes)} episodes)\033[0m")

        # Load environment and config for this task
        TASK_ENV = class_decorator(task_name)
        task_args = build_task_args(task_name, args.task_config)
        task_args["policy_name"] = args.policy_name
        task_args["save_root"] = args.save_root

        for item in episodes:
            seed = item["seed"]
            episode_idx = item["episode_idx"]
            global_idx += 1

            print(f"\n\033[33m[{global_idx}/{total_episodes}] "
                  f"task={task_name}, seed={seed}, episode={episode_idx}\033[0m")

            try:
                succ = run_one_episode(
                    TASK_ENV=TASK_ENV,
                    model=model,
                    seed=seed,
                    args=task_args,
                    save_root=str(run_dir),
                    video_guidance_scale=args.video_guidance_scale,
                    action_guidance_scale=args.action_guidance_scale,
                    save_visualization=True,
                    episode_idx=episode_idx,
                )
                results[task_name]["total"] += 1
                if succ:
                    results[task_name]["succ"] += 1
                results[task_name]["episodes"].append({
                    "seed": seed, "episode_idx": episode_idx, "success": succ
                })
            except Exception as e:
                print(f"\033[91mError in {task_name} seed={seed}: {e}\033[0m")
                traceback.print_exc()
                results[task_name]["total"] += 1
                results[task_name]["episodes"].append({
                    "seed": seed, "episode_idx": episode_idx, "success": False, "error": str(e)
                })
                # Try to close env gracefully
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass

            # Save intermediate per-client metrics
            client_metrics_file = run_dir / "metrics" / f"client_{args.client_id}.json"
            client_metrics_file.parent.mkdir(parents=True, exist_ok=True)
            client_results = {}
            for tname, res in results.items():
                client_results[tname] = {
                    "succ_num": res["succ"],
                    "total_num": res["total"],
                    "succ_rate": res["succ"] / res["total"] if res["total"] > 0 else 0.0,
                    "episodes": res["episodes"],
                }
            write_json(client_results, client_metrics_file)

            # Print progress
            print(f"  \033[96mProgress: {task_name} "
                  f"{results[task_name]['succ']}/{results[task_name]['total']} "
                  f"({results[task_name]['succ']/results[task_name]['total']*100:.1f}%)\033[0m")

        # Unload this task's environment to free GPU memory
        print(f"\033[34mUnloading task environment: {task_name}\033[0m")
        try:
            TASK_ENV.close_env()
        except Exception:
            pass
        del TASK_ENV
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\033[34mGPU memory freed after {task_name}\033[0m")

    # Final summary
    print("\n" + "=" * 60)
    print("\033[32mEvaluation Complete!\033[0m")
    print("=" * 60)
    total_succ = 0
    total_episodes = 0
    for task_name in sorted(results.keys()):
        res = results[task_name]
        rate = res["succ"] / res["total"] * 100 if res["total"] > 0 else 0
        print(f"  {task_name:<35} {res['succ']:>3}/{res['total']:<3} = {rate:.1f}%")
        total_succ += res["succ"]
        total_episodes += res["total"]

    if total_episodes > 0:
        print(f"\n  {'OVERALL':<35} {total_succ:>3}/{total_episodes:<3} "
              f"= {total_succ/total_episodes*100:.1f}%")


def merge_metrics():
    """Merge per-client metrics into final per-task results."""
    parser = argparse.ArgumentParser(description="Merge client metrics")
    parser.add_argument("--metrics_dir", type=str, required=True,
                        help="Directory containing client_*.json files")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    client_files = sorted(metrics_dir.glob("client_*.json"))

    if not client_files:
        print(f"No client metrics files in {metrics_dir}")
        sys.exit(1)

    print(f"Found {len(client_files)} client metrics files")

    # Merge all client results by task
    merged = defaultdict(lambda: {"succ": 0, "total": 0, "episodes": []})
    for cf in client_files:
        with open(cf, "r") as f:
            data = json.load(f)
        for task_name, res in data.items():
            merged[task_name]["succ"] += res["succ_num"]
            merged[task_name]["total"] += res["total_num"]
            merged[task_name]["episodes"].extend(res.get("episodes", []))

    # Write per-task final results
    print(f"\n{'Task':<35} {'Succ':<10} {'Rate':<10}")
    print("-" * 55)
    total_succ = 0
    total_episodes = 0
    for task_name in sorted(merged.keys()):
        res = merged[task_name]
        rate = res["succ"] / res["total"] if res["total"] > 0 else 0.0
        print(f"  {task_name:<35} {int(res['succ']):>3}/{int(res['total']):<3}   {rate*100:.1f}%")
        total_succ += res["succ"]
        total_episodes += res["total"]

        # Write per-task result file
        task_dir = metrics_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json({
            "succ_num": float(res["succ"]),
            "total_num": float(res["total"]),
            "succ_rate": float(rate),
            "episodes": res["episodes"],
        }, task_dir / "res.json")

    # Write overall summary
    overall_rate = total_succ / total_episodes if total_episodes > 0 else 0.0
    print(f"\n  {'OVERALL':<35} {int(total_succ):>3}/{int(total_episodes):<3}   {overall_rate*100:.1f}%")
    write_json({
        "total_succ": float(total_succ),
        "total_episodes": float(total_episodes),
        "overall_succ_rate": float(overall_rate),
    }, metrics_dir / "overall.json")

    print(f"\n\033[32mMerged results saved to {metrics_dir}/\033[0m")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        sys.argv.pop(1)
        merge_metrics()
    else:
        main()
