"""
批量跑 H2 / H3 的 3 个 seed，结果写到 experiments/logs/。

用法（在项目根目录）：
    python experiments/run_matrix.py

每个子进程都会打印很多配置；这里把 stdout 丢掉，只保留 JSON。
如果需要看某个配置的详细输出，单独运行 steps/03_plan.py。
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
PLAN = ROOT / "steps" / "03_plan.py"
LOGS = ROOT / "experiments" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]


def checkpoint_for(seed: int) -> str:
    # seed 0 的 checkpoint 没有后缀，seed 1/2 有
    if seed == 0:
        return "experiments/checkpoints/world_model_nodec.pt"
    return f"experiments/checkpoints/world_model_nodec_s{seed}.pt"


def h2_jobs():
    jobs = []
    for seed in SEEDS:
        ckpt = checkpoint_for(seed)
        for samples, horizon in [(1, 8), (8, 8), (32, 8), (128, 8), (32, 4), (32, 12)]:
            name = f"h2_n{samples}_h{horizon}_s{seed}"
            jobs.append(
                (
                    name,
                    [
                        PY, str(PLAN),
                        "--episodes", "30",
                        "--checkpoint", ckpt,
                        "--ask", "fixed",
                        "--samples", str(samples),
                        "--horizon", str(horizon),
                        "--seed", str(seed),
                        "--out", f"experiments/logs/{name}.json",
                    ],
                )
            )
    return jobs


def h3_jobs():
    # (短名, 额外参数)
    configs = [
        ("sf_never_low", ["--sign-first", "--ask", "never", "--ask-cost", "-0.05"]),
        ("sf_fixed_low", ["--sign-first", "--ask", "fixed", "--ask-cost", "-0.05"]),
        ("sf_fixed_high", ["--sign-first", "--ask", "fixed", "--ask-cost", "-0.4"]),
        ("sf_entropy08_low", ["--sign-first", "--ask", "entropy", "--threshold", "0.8", "--ask-cost", "-0.05"]),
        ("sf_entropy08_high", ["--sign-first", "--ask", "entropy", "--threshold", "0.8", "--ask-cost", "-0.4"]),
        ("sf_voi_low", ["--sign-first", "--ask", "voi", "--ask-cost", "-0.05"]),
        ("sf_voi_high", ["--sign-first", "--ask", "voi", "--ask-cost", "-0.4"]),
        ("nosf_fixed_low", ["--ask", "fixed", "--ask-cost", "-0.05"]),
        ("nosf_voi_high", ["--ask", "voi", "--ask-cost", "-0.4"]),
    ]
    jobs = []
    for seed in SEEDS:
        ckpt = checkpoint_for(seed)
        for tag, extra in configs:
            name = f"h3_{tag}_s{seed}"
            jobs.append(
                (
                    name,
                    [
                        PY, str(PLAN),
                        "--episodes", "50",
                        "--checkpoint", ckpt,
                        "--seed", str(seed),
                        "--out", f"experiments/logs/{name}.json",
                        *extra,
                    ],
                )
            )
    return jobs


def run_one(job):
    name, cmd = job
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return name, False, time.time() - t0, proc.stderr[-2000:]
    return name, True, time.time() - t0, ""


def main():
    jobs = h2_jobs() + h3_jobs()
    print(f"共 {len(jobs)} 个评估任务，2 个进程并行。")
    failed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for name, ok, elapsed, err in pool.map(run_one, jobs):
            status = "ok" if ok else "FAIL"
            print(f"[{status:4s}] {name:24s} {elapsed:5.1f}s")
            if not ok:
                failed.append((name, err))

    if failed:
        print("\n失败任务：")
        for name, err in failed:
            print(f"--- {name} ---\n{err}")
        sys.exit(1)

    print("\n全部完成。下一步：python experiments/summarize.py")


if __name__ == "__main__":
    main()
