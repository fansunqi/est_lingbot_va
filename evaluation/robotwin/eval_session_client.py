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

robowin_root = Path(os.environ.get("ROBOTWIN_ROOT", "/apdcephfs_cq8/share_1611098/stevefan/robotics/RoboTwin"))
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
import gc

from evaluation.robotwin.geometry import euler2quat
from scipy.spatial.transform import Rotation as R
from description.utils.generate_episode_instructions import *

import imageio

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
from evaluation.robotwin.test_render import Sapien_TEST


def _call_safely(obj, method_name, *args, **kwargs):
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    try:
        return method(*args, **kwargs)
    except TypeError:
        if args or kwargs:
            try:
                return method()
            except Exception:
                return None
        return None
    except Exception:
        return None


def _close_eval_video_ffmpeg(task_env):
    ffmpeg = getattr(task_env, "eval_video_ffmpeg", None)
    if ffmpeg is None:
        return

    stdin = getattr(ffmpeg, "stdin", None)
    if stdin is not None:
        try:
            stdin.close()
        except Exception:
            pass

    try:
        ffmpeg.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            ffmpeg.terminate()
            ffmpeg.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                ffmpeg.kill()
                ffmpeg.wait(timeout=2)
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass

    try:
        task_env.eval_video_ffmpeg = None
    except Exception:
        pass


def _clear_sapien_render_cache():
    try:
        from sapien.render import clear_cache as sapien_clear_cache
        sapien_clear_cache()
    except Exception:
        pass


def cleanup_env_resources(task_env, clear_cache=True):
    """Best-effort cleanup for SAPIEN scene/renderer resources after an episode."""
    if task_env is not None:
        _close_eval_video_ffmpeg(task_env)

        viewer = getattr(task_env, "viewer", None)
        if viewer is not None:
            _call_safely(viewer, "close")

        close_env = getattr(task_env, "close_env", None)
        if callable(close_env):
            try:
                close_env(clear_cache=False)
            except TypeError:
                _call_safely(task_env, "close_env")
            except Exception:
                pass
        else:
            _call_safely(task_env, "close")

        scene = getattr(task_env, "scene", None)
        if scene is not None:
            _call_safely(scene, "clear")

        cameras = getattr(task_env, "cameras", None)
        if cameras is not None:
            for attr in (
                "left_camera", "right_camera", "observer_camera",
                "world_camera1", "world_camera2", "static_camera_list",
                "static_camera_config", "static_camera_name",
            ):
                if hasattr(cameras, attr):
                    try:
                        setattr(cameras, attr, None)
                    except Exception:
                        pass
            try:
                cameras.__dict__.clear()
            except Exception:
                pass

        robot = getattr(task_env, "robot", None)
        if robot is not None:
            try:
                robot.__dict__.clear()
            except Exception:
                pass

        for resource in (
            getattr(task_env, "renderer", None),
            getattr(task_env, "engine", None),
        ):
            if resource is not None:
                _call_safely(resource, "close")
                _call_safely(resource, "destroy")
                _call_safely(resource, "release")

        for attr in (
            "cameras", "head_camera", "left_camera", "right_camera",
            "observer_camera", "world_camera1", "world_camera2",
            "scene", "renderer", "engine", "viewer", "robot",
            "eval_video_ffmpeg", "now_obs", "world_pcd", "raw_head_pcl",
            "real_head_pcl", "real_head_pcl_color", "point_light_lst",
            "cluttered_objs", "record_cluttered_objects",
        ):
            if hasattr(task_env, attr):
                try:
                    setattr(task_env, attr, None)
                except Exception:
                    pass

        try:
            task_env.__dict__.clear()
        except Exception:
            pass

    gc.collect()
    if clear_cache:
        _clear_sapien_render_cache()
        gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def run_render_probe():
    probe = None
    try:
        probe = Sapien_TEST()
    finally:
        cleanup_env_resources(probe, clear_cache=True)


def write_json(data: dict, fpath: Path) -> None:
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=2)


def group_assignment_by_task(assignment):
    """Group assignment entries by task while preserving first-seen task order."""
    grouped = {}
    for item in assignment:
        grouped.setdefault(item["task"], []).append(item)
    return list(grouped.items())


def client_metrics_path(run_dir: Path, client_id: int) -> Path:
    return Path(run_dir) / "metrics" / f"client_{client_id}.json"


def load_client_results(run_dir: Path, client_id: int):
    results = defaultdict(lambda: {"succ": 0, "total": 0, "episodes": []})
    metrics_file = client_metrics_path(run_dir, client_id)
    if not metrics_file.is_file():
        return results

    with open(metrics_file, "r") as f:
        data = json.load(f)

    for task_name, res in data.items():
        results[task_name]["succ"] = int(res.get("succ_num", 0))
        results[task_name]["total"] = int(res.get("total_num", 0))
        results[task_name]["episodes"] = list(res.get("episodes", []))
    return results


def save_client_results(run_dir: Path, client_id: int, results) -> None:
    client_results = {}
    for tname, res in results.items():
        client_results[tname] = {
            "succ_num": res["succ"],
            "total_num": res["total"],
            "succ_rate": res["succ"] / res["total"] if res["total"] > 0 else 0.0,
            "episodes": res["episodes"],
        }
    write_json(client_results, client_metrics_path(run_dir, client_id))


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


def run_one_episode(TASK_ENV, model, seed, args, save_root, episode_info,
                    video_guidance_scale=5.0, action_guidance_scale=5.0,
                    save_visualization=True, episode_idx=0):
    """
    Run a single evaluation episode for a given (task, seed).

    Returns:
        bool: whether the episode was successful
    """
    task_name = args['task_name']
    instruction_type = 'seen'

    if episode_info is None:
        raise RuntimeError(
            "Missing cached episode_info in assignment. Regenerate valid_seeds.json "
            "with evaluation.robotwin.collect_seeds before running session eval."
        )

    if isinstance(episode_info, dict) and "info" in episode_info:
        episode_info = episode_info["info"]

    TASK_ENV.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    episode_info_list = [episode_info]
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

    cleanup_env_resources(TASK_ENV, clear_cache=True)
    del full_obs_list, gen_video_list, full_action_history
    del initial_obs, initial_formatted_obs, first_obs

    if succ:
        print(f"\033[92m[{task_name} seed={seed}] Success!\033[0m")
    else:
        print(f"\033[91m[{task_name} seed={seed}] Fail!\033[0m")

    return succ


def run_task_subprocesses(args, task_groups, run_dir: Path) -> None:
    """Run each task group in a fresh Python process to release SAPIEN resources."""
    lingbot_root = Path(__file__).resolve().parents[2]
    split_dir = run_dir / "task_assignments_by_task" / f"client_{args.client_id}"
    split_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = client_metrics_path(run_dir, args.client_id)
    if not args.resume_metrics and metrics_file.exists():
        metrics_file.unlink()

    total_episodes = sum(len(episodes) for _, episodes in task_groups)
    print(f"\033[34mRestart-per-task mode enabled: {len(task_groups)} task processes, "
          f"{total_episodes} episodes total\033[0m")

    for task_idx, (task_name, episodes) in enumerate(task_groups):
        task_assignment = split_dir / f"{task_idx:03d}_{task_name}.json"
        write_json(episodes, task_assignment)

        cmd = [
            sys.executable, "-u", "-m", "evaluation.robotwin.eval_session_client",
            "--config", args.config,
            "--assignment", str(task_assignment),
            "--port", str(args.port),
            "--save_root", args.save_root,
            "--run_dir", str(run_dir),
            "--seed", str(args.seed),
            "--client_id", str(args.client_id),
            "--task_config", args.task_config,
            "--video_guidance_scale", str(args.video_guidance_scale),
            "--action_guidance_scale", str(args.action_guidance_scale),
            "--policy_name", args.policy_name,
            "--resume_metrics",
        ]

        print(f"\n\033[34m[{task_idx + 1}/{len(task_groups)}] "
              f"Starting fresh process for {task_name} ({len(episodes)} episodes)\033[0m")
        completed = subprocess.run(cmd, cwd=str(lingbot_root), env=os.environ.copy())
        if completed.returncode != 0:
            raise SystemExit(
                f"Task subprocess failed for {task_name} with exit code {completed.returncode}"
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n\033[32mAll task subprocesses completed.\033[0m")


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
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Exact run directory for metrics and visualization")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed multiplier (for stseed folder naming)")
    parser.add_argument("--client_id", type=int, default=0,
                        help="Client ID (for per-client metrics file naming)")
    parser.add_argument("--task_config", type=str, default="demo_clean",
                        help="Task config name")
    parser.add_argument("--video_guidance_scale", type=float, default=5.0)
    parser.add_argument("--action_guidance_scale", type=float, default=5.0)
    parser.add_argument("--policy_name", type=str, default="ACT")
    parser.add_argument("--restart_per_task", action="store_true",
                        help="Run each task group in a fresh child process")
    parser.add_argument("--resume_metrics", action="store_true",
                        help="Load and append to metrics/client_<id>.json before running")
    args = parser.parse_args()

    # Load assignment
    with open(args.assignment, "r") as f:
        assignment = json.load(f)

    print(f"\033[32mLoaded assignment: {len(assignment)} episodes\033[0m")

    # Group assignment by task.
    task_groups = group_assignment_by_task(assignment)

    print(f"\033[32mTasks involved ({len(task_groups)}): "
          f"{[f'{name}({len(eps)})' for name, eps in task_groups]}\033[0m")

    # All outputs go under run_dir when provided, otherwise save_root/stseed-{st_seed}/.
    st_seed = 10000 * (1 + args.seed)
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.save_root) / f"stseed-{st_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.restart_per_task:
        run_task_subprocesses(args, task_groups, run_dir)
        return

    # Initialize renderer
    run_render_probe()

    # Connect to server. The server assigns this websocket connection a session_id.
    # In restart-per-task mode each child process gets a new session, and process
    # exit closes the connection so the server can free that session's KV cache.
    model = WebsocketClientPolicy(port=args.port)
    print(f"\033[32mConnected to server on port {args.port}\033[0m")

    # Execute assignment task-by-task: load env → run all episodes → unload env
    results = load_client_results(run_dir, args.client_id) if args.resume_metrics else defaultdict(
        lambda: {"succ": 0, "total": 0, "episodes": []}
    )
    if args.resume_metrics:
        resumed_total = sum(res["total"] for res in results.values())
        print(f"\033[32mResumed metrics for client {args.client_id}: "
              f"{resumed_total} previous episodes\033[0m")

    global_idx = 0
    total_episodes = sum(len(eps) for _, eps in task_groups)

    for task_name, episodes in task_groups:
        print(f"\n\033[34m{'='*60}\033[0m")
        print(f"\033[34mLoading task environment: {task_name} ({len(episodes)} episodes)\033[0m")

        task_args = build_task_args(task_name, args.task_config)
        task_args["policy_name"] = args.policy_name
        task_args["save_root"] = args.save_root

        for item in episodes:
            TASK_ENV = None
            seed = item["seed"]
            episode_idx = item["episode_idx"]
            global_idx += 1

            print(f"\n\033[33m[{global_idx}/{total_episodes}] "
                  f"task={task_name}, seed={seed}, episode={episode_idx}\033[0m")

            try:
                TASK_ENV = class_decorator(task_name)
                succ = run_one_episode(
                    TASK_ENV=TASK_ENV,
                    model=model,
                    seed=seed,
                    args=task_args,
                    save_root=str(run_dir),
                    episode_info=item.get("episode_info"),
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
                cleanup_env_resources(TASK_ENV, clear_cache=True)
            finally:
                if TASK_ENV is not None:
                    cleanup_env_resources(TASK_ENV, clear_cache=True)
                    del TASK_ENV

            # Save intermediate per-client metrics
            save_client_results(run_dir, args.client_id, results)

            # Print progress
            print(f"  \033[96mProgress: {task_name} "
                  f"{results[task_name]['succ']}/{results[task_name]['total']} "
                  f"({results[task_name]['succ']/results[task_name]['total']*100:.1f}%)\033[0m")

        # Unload this task's environment to free GPU memory
        print(f"\033[34mUnloading task environment: {task_name}\033[0m")
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

    model.close()


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
