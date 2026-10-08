"""Background campaign supervisor for the four formal RoboTwin tasks."""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import time

POLICY = Path("/home/hanjinwei/p1/project/RoboTwin/XPolicyLab/policy/DP")
ROBOTWIN = Path("/home/hanjinwei/p1/project/RoboTwin")
EXP = POLICY / "experiments" / "robotwin2_adaptive_cnn"
CAMPAIGN = "robotwin2_adaptdiff_20261008"
ROOT = EXP / "runs" / CAMPAIGN
PYTHON = "/home/hanjinwei/p1/envs/robotwin-sc/bin/python"
TASKS = ("beat_block_hammer", "handover_block", "stack_bowls_three", "pick_dual_bottles")
MODES = (("vanilla", "0"), ("adaptive", "1"))
STATUS = ROOT / "campaign_events.jsonl"


def event(kind, **details):
    item = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": kind, **details}
    with STATUS.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def out_dir(task, mode):
    return ROOT / task / mode


def alive(pid):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError):
        return False
    return b"experiments.robotwin2_adaptive_cnn.train" in command


def complete(task, mode):
    path = out_dir(task, mode) / "status.json"
    if not path.is_file():
        return False
    return json.loads(path.read_text()).get("state") == "training_complete"


def launch(task, mode, gpu, resume=False):
    output = out_dir(task, mode)
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
    command = [
        PYTHON, "-u", "-m", "experiments.robotwin2_adaptive_cnn.train",
        "--mode", mode, "--task", task, "--run-name", CAMPAIGN, "--device", "cuda:0",
    ]
    if resume:
        command.append("--resume")
    with (output / "stdout.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=POLICY, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (output / "train.pid").write_text(str(process.pid) + "\n")
    event("training_started", task=task, mode=mode, gpu=gpu, pid=process.pid, resume=resume)
    return process.pid


def supervise_task(task, first=False):
    states = {}
    for mode, gpu in MODES:
        output = out_dir(task, mode)
        if complete(task, mode):
            states[mode] = {"state": "complete"}
            continue
        if first:
            pid = int((output / "train.pid").read_text())
            if alive(pid):
                states[mode] = {"state": "running", "pid": pid, "gpu": gpu, "retries": 0}
                continue
        pid = launch(task, mode, gpu)
        states[mode] = {"state": "running", "pid": pid, "gpu": gpu, "retries": 0}
    while any(item["state"] == "running" for item in states.values()):
        time.sleep(60)
        for mode, item in states.items():
            if item["state"] != "running" or alive(item["pid"]):
                continue
            if complete(task, mode):
                item["state"] = "complete"
                event("training_complete", task=task, mode=mode, checkpoint=str(out_dir(task, mode) / "checkpoints" / "600.ckpt"))
                continue
            output = out_dir(task, mode)
            recoverable = (output / "checkpoints" / "latest.ckpt").is_file() and (
                output / "checkpoints" / "trainer_state.pt").is_file()
            if recoverable and item["retries"] < 2:
                item["retries"] += 1
                item["pid"] = launch(task, mode, item["gpu"], resume=True)
                event("training_recovery", task=task, mode=mode, attempt=item["retries"])
            else:
                item["state"] = "failed"
                (output / "failed.txt").write_text("Training exited before completion; inspect stdout.log.\n")
                event("training_failed", task=task, mode=mode, recoverable=recoverable)
    return states


def evaluation_links(task, mode):
    output = out_dir(task, mode)
    ckpt_name = f"{task}_adaptdiff_{mode}_{CAMPAIGN}"
    link = POLICY / "checkpoints" / f"RoboTwin-{ckpt_name}-aloha_agilex-joint-42"
    target = output / "checkpoints"
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise FileExistsError(link)
    elif link.exists():
        raise FileExistsError(link)
    else:
        link.symlink_to(target, target_is_directory=True)
    eval_target = output / "eval"
    eval_target.mkdir(parents=True, exist_ok=True)
    eval_link = ROBOTWIN / "eval_result" / task / "DP" / "demo_clean" / ckpt_name
    eval_link.parent.mkdir(parents=True, exist_ok=True)
    if eval_link.is_symlink():
        if eval_link.resolve() != eval_target.resolve():
            raise FileExistsError(eval_link)
    elif eval_link.exists():
        raise FileExistsError(eval_link)
    else:
        eval_link.symlink_to(eval_target, target_is_directory=True)
    return ckpt_name


def evaluate(task, mode):
    output = out_dir(task, mode)
    ckpt_name = evaluation_links(task, mode)
    args_file = output / "eval_args.txt"
    args_file.write_text("--task_config\ndemo_clean\n--test_num\n100\n--instruction_type\nseen\n")
    env = os.environ.copy()
    env["PATH"] = "/home/hanjinwei/miniconda3/bin:" + env.get("PATH", "")
    env["ROBOTWIN_EVAL_ARGS_FILE"] = str(args_file)
    env["EVAL_ENV_TYPE"] = "sim"
    command = [
        "bash", str(POLICY / "eval.sh"), "RoboTwin", task, ckpt_name,
        "aloha_agilex", "joint", "42", "0", "1",
        "/home/hanjinwei/p1/envs/robotwin-sc",
        "/home/hanjinwei/p1/envs/robotwin-sc",
    ]
    event("evaluation_started", task=task, mode=mode, ckpt_name=ckpt_name)
    with (output / "eval.log").open("a", encoding="utf-8") as log:
        status = subprocess.run(command, cwd=POLICY, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT).returncode
    result_files = sorted((output / "eval").rglob("_result.txt"))
    if status != 0 or not result_files:
        event("evaluation_failed", task=task, mode=mode, exit_code=status)
        return None
    result_file = result_files[-1]
    result_text = result_file.read_text(errors="replace")
    match = re.search(r"Success(?:es| num)?\s*[:=]\s*(\d+)\s*/\s*(\d+)", result_text, re.I)
    if match:
        success = int(match.group(1)) / int(match.group(2))
    else:
        lines = [line.strip() for line in result_text.splitlines() if line.strip()]
        try:
            success = float(lines[-1])
        except ValueError:
            success = None
    result = {"task": task, "mode": mode, "success_rate": success,
              "result_file": str(result_file), "checkpoint": str(output / "checkpoints" / "600.ckpt")}
    (output / "evaluation_result.json").write_text(json.dumps(result, indent=2))
    event("evaluation_complete", **result)
    return result


def publish(results, states):
    evaluated = {(item["task"], item["mode"]): item for item in results}
    records = []
    curves = {}
    for task in TASKS:
        for mode, _ in MODES:
            output = out_dir(task, mode)
            log_path = output / "logs.jsonl"
            rows = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.is_file() else []
            evaluation = evaluated.get((task, mode), {})
            record = {
                "task": task,
                "mode": mode,
                "training_state": states[task][mode]["state"],
                "epochs_completed": rows[-1]["epoch"] if rows else 0,
                "training_wall_seconds": sum(row["epoch_seconds"] for row in rows),
                "optimizer_seconds": rows[-1]["train_seconds"] if rows else None,
                "final_train_loss": rows[-1]["train_loss"] if rows else None,
                "final_uniform_val_loss": rows[-1]["uniform_val_loss"] if rows else None,
                "success_rate": evaluation.get("success_rate"),
                "checkpoint": str(output / "checkpoints" / "600.ckpt") if complete(task, mode) else None,
                "result_file": evaluation.get("result_file"),
            }
            records.append(record)
            curves[f"{task}_{mode}"] = rows
    differences = {}
    for task in TASKS:
        pair = [record for record in records if record["task"] == task]
        if all(record["success_rate"] is not None for record in pair):
            differences[task] = pair[1]["success_rate"] - pair[0]["success_rate"]
    all_complete = len(results) == 8 and all(
        states[task][mode]["state"] == "complete"
        for task in TASKS for mode, _ in MODES)
    summary = {"campaign": CAMPAIGN, "complete": all_complete,
               "records": records, "adaptive_minus_vanilla": differences}
    (ROOT / "results.json").write_text(json.dumps(summary, indent=2))
    repo = Path("/home/hanjinwei/p1/project/adapt-diff")
    dest = repo / "experiments" / "robotwin2_adaptive_cnn" / "results"
    dest.mkdir(parents=True, exist_ok=True)
    suffix = "" if all_complete else "_partial"
    (dest / f"{CAMPAIGN}{suffix}.json").write_text(json.dumps(summary, indent=2))
    curves_dir = dest / f"{CAMPAIGN}_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    curve_fields = ("epoch", "global_step", "train_loss", "uniform_val_loss",
                    "epoch_seconds", "train_seconds", "mean_batch_seconds",
                    "sampler_updates", "optimizer_steps")
    for name, rows in curves.items():
        with (curves_dir / f"{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=curve_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in curve_fields})
    subprocess.run(["git", "-C", str(repo), "add", str(dest)], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m",
                    f"Record {CAMPAIGN} RoboTwin results"], check=True)
    subprocess.run(["git", "-C", str(repo), "push", "origin", "main"], check=True)
    event("results_published", repo=str(repo), complete=all_complete)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    states = {}
    for index, task in enumerate(TASKS):
        event("task_pair_started", task=task)
        states[task] = supervise_task(task, first=(index == 0))
        event("task_pair_finished", task=task,
              vanilla=states[task]["vanilla"]["state"],
              adaptive=states[task]["adaptive"]["state"])
    results = []
    for task in TASKS:
        for mode, _ in MODES:
            if states[task][mode]["state"] == "complete":
                result = evaluate(task, mode)
                if result:
                    results.append(result)
    publish(results, states)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        event("campaign_error", error=repr(error))
        raise

