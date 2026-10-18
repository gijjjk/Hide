"""
第 2 步：用世界模型在潜空间里“思考”（规划）。

先跑默认设置：
    python steps/03_plan.py --episodes 100 --horizon 8 --samples 32

对比“思考预算”（N 越大，想得越久）：
    python steps/03_plan.py --samples 1
    python steps/03_plan.py --samples 32
    python steps/03_plan.py --samples 128

对比询问策略（第 3 步的预告）：
    python steps/03_plan.py --ask never
    python steps/03_plan.py --ask fixed
    python steps/03_plan.py --ask entropy --threshold 0.5

冒烟测试：
    python steps/03_plan.py --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from askdreamer.controller import SignFirstController, make_ask_policy
from askdreamer.data import ScriptedPolicy
from askdreamer.env import ACTION_NAMES, N_ACTIONS, HiddenGoalGrid
from askdreamer.model import WorldModel
from askdreamer.planner import RandomShootingPlanner
from askdreamer.train import save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--policy", type=str, default="planner", choices=["planner", "random", "scripted"])
    p.add_argument("--ask", type=str, default="never", choices=["never", "fixed", "entropy", "voi"])
    p.add_argument("--threshold", type=float, default=0.5, help="熵门控阈值（nat）")
    p.add_argument("--ask-after", type=int, default=0, help="至少走多少步后才允许询问")
    p.add_argument("--sign-first", action="store_true", help="先读告示牌，再决定是否询问")
    p.add_argument("--ask-cost", type=float, default=-0.05, help="询问的奖励代价（负数）")
    p.add_argument("--wrong-penalty", type=float, default=-0.3, help="走错门的奖励代价（负数）")
    p.add_argument("--horizon", type=int, default=8, help="规划视野 H")
    p.add_argument("--samples", type=int, default=32, help="每次规划的候选序列数 N")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--no-value", action="store_true", help="规划时不用价值头（只用奖励头）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ask-prob", type=float, default=0.5, help="scripted 基线的询问概率")
    p.add_argument("--sign-prob", type=float, default=0.3, help="scripted 基线去读告示牌的概率")
    p.add_argument("--checkpoint", type=str, default="experiments/checkpoints/world_model.pt")
    p.add_argument("--out", type=str, default="experiments/logs/plan_metrics.json")
    p.add_argument("--render", action="store_true", help="打印每局结束时的 ASCII 地图")
    p.add_argument("--verbose", action="store_true", help="打印每局结果")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def load_world_model(path: str) -> WorldModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = WorldModel(**ckpt["config"])
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"警告：checkpoint 缺少 {len(missing)} 个参数（旧版本？value_head 会随机初始化）")
        print("      建议用当前代码重新训练：python steps/02_world_model.py ...")
    model.eval()
    return model


@torch.no_grad()
def run_planner_episode(env, model, planner, ask_policy, seed: int):
    """跑一局：状态更新 -> 决定问不问 -> 规划移动 -> 和环境交互。"""
    obs, msg = env.reset(seed=seed)

    # 初始潜状态：用当前观测和消息做一次 posterior 推断
    h = torch.zeros(1, model.h_dim)
    z = torch.zeros(1, model.z_dim)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)   # (1,150)
    msg_t = torch.tensor(msg, dtype=torch.float32).unsqueeze(0)   # (1,6)
    h, z, _, _ = model.observe_step(h, z, None, msg_t, obs_t)

    total_reward = 0.0
    planner_ms = 0.0
    planner_calls = 0
    done = False

    while not done:
        # 0) 某些控制器会先执行一个探索动作（例如先读告示牌）
        action = None
        if hasattr(ask_policy, "override_action"):
            action = ask_policy.override_action(env)

        # 1) 决定是否主动询问
        if action is not None:
            pass
        elif ask_policy.should_ask(model, h, z, env):
            action = 4
        else:
            # 2) 在潜空间里想象，选第一个动作
            t0 = time.time()
            action, _ = planner.act(h, z, msg_t)
            planner_ms += (time.time() - t0) * 1000.0
            planner_calls += 1

        # 3) 和真实环境交互
        obs, msg, reward, done, info = env.step(action)
        total_reward += reward

        # 4) 用新观测/新消息更新潜状态（动作是刚刚执行的）
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        msg_t = torch.tensor(msg, dtype=torch.float32).unsqueeze(0)
        h, z, _, _ = model.observe_step(h, z, torch.tensor([action]), msg_t, obs_t)

    return {
        "return": total_reward,
        "steps": env.steps,
        "success": bool(info["success"]),
        "ask_count": int(info["ask_count"]),
        "sign_read": bool(info["sign_read"]),
        "planner_ms": planner_ms / max(1, planner_calls),
        "final_info": info,
    }


def run_random_episode(env, rng: np.random.Generator, seed: int):
    obs, msg = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        action = int(rng.integers(0, N_ACTIONS))
        obs, msg, reward, done, info = env.step(action)
        total_reward += reward
    return {
        "return": total_reward,
        "steps": env.steps,
        "success": bool(info["success"]),
        "ask_count": int(info["ask_count"]),
        "sign_read": bool(info["sign_read"]),
        "planner_ms": 0.0,
        "final_info": info,
    }


def run_scripted_episode(env, rng: np.random.Generator, seed: int, ask_prob: float, sign_prob: float):
    obs, msg = env.reset(seed=seed)
    policy = ScriptedPolicy(rng, ask_prob=ask_prob, sign_prob=sign_prob)
    total_reward = 0.0
    done = False
    while not done:
        action = policy.act(env)
        obs, msg, reward, done, info = env.step(action)
        total_reward += reward
    return {
        "return": total_reward,
        "steps": env.steps,
        "success": bool(info["success"]),
        "ask_count": int(info["ask_count"]),
        "sign_read": bool(info["sign_read"]),
        "planner_ms": 0.0,
        "final_info": info,
    }


def main():
    args = parse_args()

    if args.smoke:
        args.episodes = 3
        args.samples = 8
        args.horizon = 4

    torch.set_num_threads(3)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("配置：")
    for key, value in vars(args).items():
        print(f"  {key:14s} = {value}")
    print("=" * 60)

    env = HiddenGoalGrid(
        seed=args.seed,
        ask_cost=args.ask_cost,
        wrong_penalty=args.wrong_penalty,
    )
    model = None
    planner = None
    ask_policy = None

    if args.policy == "planner":
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"\n找不到 checkpoint：{ckpt_path.resolve()}")
            print("请先运行：python steps/02_world_model.py --episodes 400 --steps 2000")
            sys.exit(1)

        model = load_world_model(str(ckpt_path))
        planner = RandomShootingPlanner(
            model,
            horizon=args.horizon,
            samples=args.samples,
            gamma=args.gamma,
            use_value=not args.no_value,
        )
        ask_policy = make_ask_policy(args.ask, args.threshold, args.ask_after)
        if args.sign_first:
            ask_policy = SignFirstController(ask_policy)
        print(f"\n已加载世界模型：{ckpt_path.resolve()}")
        print(f"规划器：H={args.horizon}, N={args.samples}, gamma={args.gamma}")
        print(f"询问策略：{ask_policy!r}")
    else:
        print(f"\n基线策略：{args.policy}")

    results = []
    t0 = time.time()

    for ep in range(args.episodes):
        seed = args.seed * 100_000 + ep

        if args.policy == "planner":
            result = run_planner_episode(env, model, planner, ask_policy, seed)
        elif args.policy == "random":
            result = run_random_episode(env, np.random.default_rng(seed + 999), seed)
        else:
            result = run_scripted_episode(
                env,
                np.random.default_rng(seed + 999),
                seed,
                ask_prob=args.ask_prob,
                sign_prob=args.sign_prob,
            )

        result["episode"] = ep
        result.pop("final_info", None)   # JSON 里不存 numpy 数组
        results.append(result)

        if args.verbose:
            print(
                f"ep {ep:3d} | return {result['return']:6.3f} | "
                f"steps {result['steps']:2d} | success {int(result['success'])} | "
                f"ask {result['ask_count']} | sign {int(result['sign_read'])}"
            )

        if args.render:
            print(f"\n--- 第 {ep} 局结束 ---")
            print(env.render(show_goal=True))
            print()

    total_sec = time.time() - t0
    n = len(results)
    success_rate = sum(r["success"] for r in results) / n
    avg_return = sum(r["return"] for r in results) / n
    avg_steps = sum(r["steps"] for r in results) / n
    avg_ask = sum(r["ask_count"] for r in results) / n
    ask_rate = sum(1 for r in results if r["ask_count"] > 0) / n
    sign_rate = sum(1 for r in results if r["sign_read"]) / n
    avg_planner_ms = sum(r["planner_ms"] for r in results if r["planner_ms"] > 0)
    n_planner = sum(1 for r in results if r["planner_ms"] > 0)
    avg_planner_ms = avg_planner_ms / max(1, n_planner)

    summary = {
        "args": vars(args),
        "episodes": n,
        "success_rate": success_rate,
        "avg_return": avg_return,
        "avg_steps": avg_steps,
        "ask_rate": ask_rate,
        "avg_ask_count": avg_ask,
        "sign_rate": sign_rate,
        "avg_planner_ms_per_step": avg_planner_ms,
        "total_seconds": round(total_sec, 2),
        "per_episode": results,
    }

    print("\n" + "=" * 60)
    print(f"策略 = {args.policy}" + (f" / 询问 = {args.ask}" if args.policy == "planner" else ""))
    print(f"局数              = {n}")
    print(f"成功率            = {success_rate:.3f}")
    print(f"平均回报          = {avg_return:.3f}")
    print(f"平均步数          = {avg_steps:.1f}")
    print(f"询问率（至少问一次）= {ask_rate:.3f}")
    print(f"平均询问次数      = {avg_ask:.2f}")
    print(f"读告示牌比例      = {sign_rate:.3f}")
    if avg_planner_ms > 0:
        print(f"规划单步耗时      = {avg_planner_ms:.2f} ms")
    print(f"总耗时            = {total_sec:.1f} 秒")
    print("=" * 60)

    save_json(args.out, summary)
    print(f"已保存指标：{Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
