#!/usr/bin/env python3
"""
Balance task assignment across GPUs based on step limits.

Uses a greedy bin-packing algorithm: assign each task (sorted by descending
step_lim) to the GPU with the currently lowest total load. This minimizes
the maximum load across GPUs (i.e., the overall evaluation wall-clock time).

Usage:
    python -m evaluation.robotwin.balance_tasks --num_clients 6
    python -m evaluation.robotwin.balance_tasks --num_clients 9 --tasks "task1 task2 ..."
"""

import argparse
import heapq

# Default: all 50 RoboTwin evaluation tasks
ALL_TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller",
    "handover_block", "handover_mic", "hanging_mug", "lift_pot",
    "move_can_pot", "move_playingcard_away", "move_pillbottle_pad", "move_stapler_pad",
    "open_laptop", "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
    "place_a2b_left", "place_a2b_right", "place_bread_basket", "place_bread_skillet",
    "place_burger_fries", "place_can_basket", "place_cans_plasticbox",
    "place_container_plate", "place_dual_shoes", "place_empty_cup", "place_fan",
    "place_mouse_pad", "place_object_basket", "place_object_scale", "place_object_stand",
    "place_phone_stand", "place_shoe", "press_stapler", "put_bottles_dustbin",
    "put_object_cabinet", "rotate_qrcode", "scan_object", "shake_bottle",
    "shake_bottle_horizontally", "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch",
]

# Step limits from RoboTwin task_config/_eval_step_limit.yml
STEP_LIMITS = {
    "adjust_bottle": 400,
    "beat_block_hammer": 400,
    "blocks_ranking_rgb": 1200,
    "blocks_ranking_size": 1200,
    "click_alarmclock": 400,
    "click_bell": 400,
    "dump_bin_bigbin": 600,
    "grab_roller": 400,
    "handover_block": 800,
    "handover_mic": 600,
    "hanging_mug": 900,
    "lift_pot": 400,
    "move_can_pot": 400,
    "move_playingcard_away": 400,
    "move_pillbottle_pad": 400,
    "move_stapler_pad": 400,
    "open_laptop": 700,
    "open_microwave": 1500,
    "pick_diverse_bottles": 400,
    "pick_dual_bottles": 400,
    "place_a2b_left": 400,
    "place_a2b_right": 400,
    "place_bread_basket": 700,
    "place_bread_skillet": 500,
    "place_burger_fries": 500,
    "place_can_basket": 700,
    "place_cans_plasticbox": 800,
    "place_container_plate": 400,
    "place_dual_shoes": 600,
    "place_empty_cup": 500,
    "place_fan": 400,
    "place_mouse_pad": 400,
    "place_object_basket": 700,
    "place_object_scale": 400,
    "place_object_stand": 400,
    "place_phone_stand": 400,
    "place_shoe": 500,
    "press_stapler": 400,
    "put_bottles_dustbin": 1700,
    "put_object_cabinet": 700,
    "rotate_qrcode": 400,
    "scan_object": 500,
    "shake_bottle": 700,
    "shake_bottle_horizontally": 700,
    "stack_blocks_three": 1200,
    "stack_blocks_two": 800,
    "stack_bowls_three": 1200,
    "stack_bowls_two": 900,
    "stamp_seal": 400,
    "turn_switch": 400,
}


def normalize_seed_entry(task_name: str, entry, episode_idx: int) -> dict:
    if isinstance(entry, int):
        raise ValueError(
            f"{task_name} seed {entry} is using the legacy valid_seeds format "
            "without episode_info. Regenerate valid_seeds.json with collect_seeds.py."
        )
    if isinstance(entry, dict) and "seed" in entry:
        if "episode_info" not in entry:
            raise ValueError(
                f"{task_name} seed {entry['seed']} is missing cached episode_info. "
                "Regenerate valid_seeds.json with collect_seeds.py."
            )
        item = {
            "task": task_name,
            "seed": entry["seed"],
            "episode_idx": entry.get("episode_idx", episode_idx),
            "episode_info": entry["episode_info"],
        }
        return item
    raise ValueError(f"Invalid seed entry for {task_name}: {entry!r}")


def balance_tasks(tasks: list[str], num_groups: int) -> list[list[str]]:
    """
    Distribute tasks into num_groups groups for parallel execution.

    Within each group, all tasks run in parallel (one client per task).
    The wall-clock time of a group = max(step_lim) in that group × test_num.

    Strategy: spread high step_lim tasks across groups evenly, then fill
    remaining slots with smaller tasks to keep group sizes balanced.

    Uses greedy LPT (Longest Processing Time first) to minimize the
    maximum total load per group. This works well because:
    - It spreads heavy tasks across groups
    - The total load correlates with wall-clock time when multiple clients
      share GPU resources (more tasks = more contention)

    Returns:
        List of num_groups lists, each containing task names for that group.
    """
    # Sort tasks by step_lim descending
    sorted_tasks = sorted(tasks, key=lambda t: STEP_LIMITS.get(t, 1000), reverse=True)

    # Min-heap: (current_load, bin_index)
    heap = [(0, i) for i in range(num_groups)]
    heapq.heapify(heap)

    bins: list[list[str]] = [[] for _ in range(num_groups)]

    for task in sorted_tasks:
        load, idx = heapq.heappop(heap)
        bins[idx].append(task)
        heapq.heappush(heap, (load + STEP_LIMITS.get(task, 1000), idx))

    return bins


def balance_sessions(valid_seeds: dict, num_clients: int, test_num: int | None = None) -> list[list[dict]]:
    """
    Distribute (task, seed) pairs across num_clients bins by step_lim.

    Each work unit is one episode: (task_name, seed) with weight = step_lim[task].
    Uses LPT greedy to minimize max total load.

    Within each bin, work units are sorted by task name to group same-task
    episodes together (reduces environment switching overhead).

    Args:
        valid_seeds: dict mapping task_name -> list of cached valid seed entries.
            Each entry must include seed and episode_info.
        num_clients: number of client bins
        test_num: optional max number of seeds to use per task. When a
            valid_seeds.json was collected for a larger eval, only the first
            test_num seeds are assigned.

    Returns:
        List of num_clients lists. Each inner list contains dicts:
        [{"task": "lift_pot", "seed": 10000}, ...]
    """
    # Build all work units: (step_lim, assignment_item)
    # episode_idx is the global index of this seed within its task
    work_units = []
    for task_name, seed_entries in valid_seeds.items():
        step_lim = STEP_LIMITS.get(task_name, 1000)
        if test_num is not None:
            seed_entries = seed_entries[:test_num]
        for idx, entry in enumerate(seed_entries):
            work_units.append((step_lim, normalize_seed_entry(task_name, entry, idx)))

    # Sort by step_lim descending (LPT)
    work_units.sort(key=lambda x: x[0], reverse=True)

    # Min-heap: (current_load, bin_index)
    heap = [(0, i) for i in range(num_clients)]
    heapq.heapify(heap)

    bins: list[list[dict]] = [[] for _ in range(num_clients)]

    for step_lim, item in work_units:
        load, idx = heapq.heappop(heap)
        bins[idx].append(item)
        heapq.heappush(heap, (load + step_lim, idx))

    # Sort each bin by task name to group same-task episodes together
    for b in bins:
        b.sort(key=lambda x: x["task"])

    return bins


def main():
    import json
    import sys

    parser = argparse.ArgumentParser(description="Balance tasks across groups by step limits")
    parser.add_argument("--mode", type=str, choices=["task", "session"], default="task",
                        help="'task': balance by task (default). 'session': balance by (task, seed) pairs")
    parser.add_argument("--num_clients", type=int, default=6,
                        help="Number of groups/clients")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Space-separated task names (default: all 50 tasks)")
    parser.add_argument("--group_id", type=int, default=None,
                        help="Output only the tasks for this group (0-indexed)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed load info to stderr")
    # Session mode args
    parser.add_argument("--valid_seeds", type=str, default=None,
                        help="[session mode] Path to valid_seeds.json")
    parser.add_argument("--output_dir", type=str, default="./task_assignments",
                        help="[session mode] Output directory for client assignment files")
    parser.add_argument("--test_num", type=int, default=None,
                        help="[session mode] Use only the first N valid seeds per task")
    args = parser.parse_args()

    if args.mode == "session":
        # Session-level balancing
        if not args.valid_seeds:
            print("Error: --valid_seeds is required in session mode", file=sys.stderr)
            sys.exit(1)

        with open(args.valid_seeds, "r") as f:
            valid_seeds = json.load(f)

        # Filter to only specified tasks if --tasks is given
        if args.tasks:
            filter_tasks = set(args.tasks.split())
            valid_seeds = {k: v for k, v in valid_seeds.items() if k in filter_tasks}

        if args.test_num is not None:
            too_short = {
                task_name: len(seed_entries)
                for task_name, seed_entries in valid_seeds.items()
                if len(seed_entries) < args.test_num
            }
            if too_short:
                details = ", ".join(f"{task}={count}" for task, count in sorted(too_short.items()))
                print(
                    f"Error: valid_seeds has fewer than --test_num={args.test_num} entries for: {details}",
                    file=sys.stderr,
                )
                sys.exit(1)

        bins = balance_sessions(valid_seeds, args.num_clients, args.test_num)

        # Output assignment files
        from pathlib import Path
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_steps = 0
        loads = []
        for i, b in enumerate(bins):
            out_file = output_dir / f"client_{i}.json"
            with open(out_file, "w") as f:
                json.dump(b, f, indent=2)
            load = sum(STEP_LIMITS.get(item["task"], 1000) for item in b)
            loads.append(load)
            total_steps += load

        if args.verbose:
            print(f"Total work units: {sum(len(b) for b in bins)}, "
                  f"Total steps: {total_steps}", file=sys.stderr)
            print(f"Ideal per client: {total_steps / args.num_clients:.0f}", file=sys.stderr)
            print(f"{'Client':<8} {'Load':<8} {'#Units':<8} {'#Tasks':<8}", file=sys.stderr)
            for i, b in enumerate(bins):
                tasks_in_bin = set(item["task"] for item in b)
                print(f"{i:<8} {loads[i]:<8} {len(b):<8} {len(tasks_in_bin):<8}", file=sys.stderr)
            print(f"Max load: {max(loads)}, Min load: {min(loads)}, "
                  f"Imbalance: {max(loads) - min(loads)} steps", file=sys.stderr)

        print(f"Saved {args.num_clients} assignment files to {args.output_dir}/")

    else:
        # Task-level balancing (original behavior)
        if args.tasks:
            tasks = args.tasks.split()
        else:
            tasks = ALL_TASKS

        bins = balance_tasks(tasks, args.num_clients)

        if args.verbose:
            total_steps = sum(STEP_LIMITS.get(t, 1000) for t in tasks)
            print(f"Total tasks: {len(tasks)}, Total steps: {total_steps}", file=sys.stderr)
            print(f"Ideal per group: {total_steps / args.num_clients:.0f}", file=sys.stderr)
            print(f"{'Group':<8} {'Load':<8} {'MaxLim':<8} {'#Tasks':<8} Tasks", file=sys.stderr)
            for i, bin_tasks in enumerate(bins):
                load = sum(STEP_LIMITS.get(t, 1000) for t in bin_tasks)
                max_lim = max(STEP_LIMITS.get(t, 1000) for t in bin_tasks) if bin_tasks else 0
                print(f"{i:<8} {load:<8} {max_lim:<8} {len(bin_tasks):<8} {' '.join(bin_tasks)}", file=sys.stderr)
            max_load = max(sum(STEP_LIMITS.get(t, 1000) for t in b) for b in bins)
            min_load = min(sum(STEP_LIMITS.get(t, 1000) for t in b) for b in bins)
            print(f"Max load: {max_load}, Min load: {min_load}, "
                  f"Imbalance: {max_load - min_load} steps", file=sys.stderr)

        if args.group_id is not None:
            if 0 <= args.group_id < len(bins):
                print(" ".join(bins[args.group_id]))
            else:
                print(f"group_id {args.group_id} out of range [0, {len(bins)-1}]", file=sys.stderr)
                sys.exit(1)
        else:
            # Output all groups, one per line
            for bin_tasks in bins:
                print(" ".join(bin_tasks))


if __name__ == "__main__":
    main()
