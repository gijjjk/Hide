"""
第 0 步：环境 + 随机策略基线。

运行：
    python steps/01_env.py --episodes 100
    python steps/01_env.py --episodes 1 --render --show-goal

这个脚本不需要 PyTorch。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 让脚本从 steps/ 目录运行时也能 import 到项目根目录的 askdreamer 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 默认可能是 GBK，这里统一成 UTF-8，避免中文乱码
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from askdreamer.env import N_ACTIONS, HiddenGoalGrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="显示第一局的 ASCII 动画")
    parser.add_argument("--show-goal", action="store_true", help="调试用：在动画里标出正确门 *")
    args = parser.parse_args()

    env = HiddenGoalGrid(seed=args.seed)
    policy_rng = np.random.default_rng(args.seed + 10_000)

    successes = 0
    returns = []
    lengths = []
    asks = 0
    total_steps = 0

    for episode in range(args.episodes):
        obs, msg = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        render_this = args.render and episode == 0

        while not done:
            action = int(policy_rng.integers(0, N_ACTIONS))
            obs, msg, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            if render_this:
                # 清屏 + 光标回到左上角（Windows 10 的终端支持 ANSI）
                print("\033[2J\033[H", end="")
                print(f"episode={episode}  step={steps}  action={action}  reward={reward:+.3f}")
                print(f"steps={env.steps}  asks={env.ask_count}  sign_read={env.sign_read}")
                print(env.render(show_goal=args.show_goal))
                print()
                time.sleep(0.15)

        successes += int(info["success"])
        returns.append(total_reward)
        lengths.append(steps)
        asks += env.ask_count
        total_steps += steps

    n = max(1, args.episodes)
    print("=" * 50)
    print(f"局数:            {args.episodes}")
    print(f"成功率:          {successes / n:.1%}")
    print(f"平均回报:        {np.mean(returns):+.3f}")
    print(f"平均步数:        {np.mean(lengths):.1f}")
    print(f"总询问次数:      {asks}")
    print(f"询问率(次/步):   {asks / max(1, total_steps):.3f}")
    print("=" * 50)
    print("提示：随机策略只是基线。后面世界模型/规划/询问都应该明显超过它。")


if __name__ == "__main__":
    main()
