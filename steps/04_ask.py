"""
第 3 步：对比“什么时候问”的策略。

它本身不重新实现规划，而是用同一套参数分别调用 steps/03_plan.py：
    never   : 从不问
    fixed   : 第一次有机会就问
    entropy : 信念熵超过阈值才问
    voi     : 信息价值 > 询问成本才问
    learned : 世界模型的 ask_head 决定是否问（需要 --learned-ask 训练的 checkpoint）

用法（在 D:\\hide 下）：
    # 低询问成本，四个策略全比
    python steps/04_ask.py --policy all --sign-first --ask-cost -0.05

    # 高询问成本
    python steps/04_ask.py --policy all --sign-first --ask-cost -0.4

    # 只跑 VOI
    python steps/04_ask.py --policy voi --sign-first --ask-cost -0.4

    # 冒烟测试
    python steps/04_ask.py --smoke
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from askdreamer.train import save_json

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "steps" / "03_plan.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--policy",
        type=str,
        default="all",
        choices=["all", "never", "fixed", "entropy", "voi", "learned"],
        help="要比较的询问策略；all = 原来的四个规则策略",
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.8, help="熵门控阈值（nat）")
    p.add_argument("--ask-cost", type=float, default=-0.05, help="询问的奖励代价（负数）")
    p.add_argument("--sign-first", action="store_true", help="先读告示牌，再决定是否询问")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/world_model_nodec.pt",
    )
    p.add_argument("--out", type=str, default="experiments/logs/ask_compare.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def run_one(policy: str, args) -> dict:
    """调用 03_plan.py 跑一个策略，返回它的 summary JSON。"""
    tmp_out = ROOT / "experiments" / "logs" / f"_ask_{policy}_tmp.json"
    tmp_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PLAN),
        "--policy", "planner",
        "--episodes", str(args.episodes),
        "--ask", policy,
        "--threshold", str(args.threshold),
        "--ask-cost", str(args.ask_cost),
        "--horizon", str(args.horizon),
        "--samples", str(args.samples),
        "--seed", str(args.seed),
        "--checkpoint", args.checkpoint,
        "--out", str(tmp_out),
    ]
    if args.sign_first:
        cmd.append("--sign-first")

    print(f"\n>>> 运行 {policy} ...")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"策略 {policy} 运行失败")

    import json

    with open(tmp_out, "r", encoding="utf-8") as f:
        data = json.load(f)
    tmp_out.unlink(missing_ok=True)
    return data


def main():
    args = parse_args()

    if args.smoke:
        args.episodes = 5
        args.samples = 8
        args.horizon = 4
        args.policy = "all"

    policies = ["never", "fixed", "entropy", "voi"] if args.policy == "all" else [args.policy]

    print("=" * 70)
    print("询问策略对比")
    print(f"  局数        = {args.episodes}")
    print(f"  询问成本    = {args.ask_cost}")
    print(f"  先读告示牌  = {args.sign_first}")
    print(f"  熵阈值      = {args.threshold}")
    print(f"  checkpoint  = {args.checkpoint}")
    print(f"  策略        = {policies}")
    print("=" * 70)

    results = {}
    for policy in policies:
        data = run_one(policy, args)
        results[policy] = {
            k: v for k, v in data.items() if k != "per_episode"
        }

    print("\n" + "=" * 78)
    header = f"{'策略':<10} {'成功率':>8} {'平均回报':>10} {'询问率':>8} {'平均询问':>8} {'平均步数':>8}"
    print(header)
    print("-" * 78)
    for policy in policies:
        d = results[policy]
        print(
            f"{policy:<10} {d['success_rate']:>8.3f} {d['avg_return']:>10.3f} "
            f"{d['ask_rate']:>8.3f} {d['avg_ask_count']:>8.2f} {d['avg_steps']:>8.1f}"
        )
    print("=" * 78)

    payload = {
        "args": vars(args),
        "policies": policies,
        "results": results,
    }
    save_json(args.out, payload)
    print(f"\n已保存：{Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
