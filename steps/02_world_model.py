"""
第 1 步：训练潜状态世界模型。

带解码器：
    python steps/02_world_model.py --episodes 400 --steps 2000

无解码器：
    python steps/02_world_model.py --episodes 400 --steps 2000 --no-decoder

学习型询问门控（在无解码器模型上加 ask_head）：
    python steps/02_world_model.py --episodes 400 --steps 2000 --no-decoder \
      --learned-ask

冒烟测试（1 分钟内）：
    python steps/02_world_model.py --smoke

启用 ABSTAIN（6 动作，训练新 checkpoint；旧 5 动作权重不兼容）：
    python steps/02_world_model.py --enable-abstain --episodes 400 --steps 2000 \
      --no-decoder

输出：
    experiments/checkpoints/world_model.pt
    experiments/logs/world_model_metrics.json
    （启用 ABSTAIN 时默认是 world_model_abstain.pt / world_model_abstain_metrics.json，
      启用 --learned-ask 时默认是 world_model_askhead.pt / world_model_askhead_metrics.json）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# 让脚本从 steps/ 目录运行时也能 import 到项目根目录的 askdreamer 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from askdreamer.data import EpisodeDataset, collect_episodes
from askdreamer.env import (
    MSG_DIM,
    N_ACTIONS,
    N_ACTIONS_WITH_ABSTAIN,
    OBS_DIM,
)
from askdreamer.model import WorldModel
from askdreamer.train import (
    count_parameters,
    evaluate_world_model,
    save_json,
    seed_everything,
    train_world_model,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=600, help="收集多少局数据")
    p.add_argument("--seq-len", type=int, default=16, help="训练序列长度")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000, help="训练步数")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--policy", type=str, default="mixed", choices=["mixed", "scripted", "random"])
    p.add_argument("--grid-size", type=int, default=7, help="环境边长（默认 7x7）")
    p.add_argument("--max-steps", type=int, default=40, help="每局最大步数")
    p.add_argument("--no-decoder", action="store_true", help="关闭观测解码器（H1 研究版本）")
    p.add_argument(
        "--enable-abstain",
        action="store_true",
        help="启用第 6 个动作 ABSTAIN（默认关闭，关闭时动作空间仍是 5，旧 checkpoint 兼容）",
    )
    p.add_argument(
        "--abstain-prob",
        type=float,
        default=0.15,
        help="脚本策略在信息不足且门口时选择 ABSTAIN 的概率",
    )
    p.add_argument(
        "--abstain-penalty", type=float, default=-0.1, help="ABSTAIN 的额外奖励代价（负数）"
    )
    p.add_argument(
        "--random-abstain-prob",
        type=float,
        default=0.05,
        help="随机策略每个非脚本步选择 ABSTAIN 的概率（仅 6 动作模式）",
    )
    p.add_argument(
        "--learned-ask",
        action="store_true",
        help="给世界模型加 ask_head，用 data.ask_label 监督学习型询问门控",
    )
    p.add_argument(
        "--ask-costs",
        type=str,
        default="-0.02,-0.05,-0.2,-0.4",
        help="训练数据中轮流使用的询问成本（逗号分隔，仅 --learned-ask 时生效）",
    )
    p.add_argument(
        "--learned-ask-weight",
        type=float,
        default=0.2,
        help="ask_head 损失权重",
    )
    p.add_argument(
        "--ask-cost-scale",
        type=float,
        default=20.0,
        help="ask_head 输入成本的缩放系数（1/0.05=20，仅 --learned-ask 时生效）",
    )
    p.add_argument("--smoke", action="store_true", help="几十步跑通流程，不追求效果")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="默认 world_model.pt；启用 ABSTAIN 时默认 world_model_abstain.pt",
    )
    p.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="默认 world_model_metrics.json；启用 ABSTAIN 时默认 world_model_abstain_metrics.json",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        args.episodes = 20
        args.seq_len = 8
        args.batch = 8
        args.steps = 30
        args.log_every = 5
        args.eval_batches = 2

    # 默认输出：启用 ABSTAIN / ask_head 时换新文件，绝不覆盖旧 checkpoint
    suffix = ""
    if args.enable_abstain:
        suffix += "_abstain"
    if args.learned_ask:
        suffix += "_askhead"
    if args.out is None:
        args.out = f"experiments/checkpoints/world_model{suffix}.pt"
    if args.metrics_out is None:
        args.metrics_out = f"experiments/logs/world_model{suffix}_metrics.json"

    ask_costs = None
    if args.learned_ask:
        ask_costs = [float(x) for x in args.ask_costs.split(",") if x.strip()]

    n_actions = N_ACTIONS_WITH_ABSTAIN if args.enable_abstain else N_ACTIONS

    # 限制 PyTorch 线程数，保留一个核给系统
    torch.set_num_threads(3)
    seed_everything(args.seed)

    print("=" * 60)
    print("配置：")
    for key, value in vars(args).items():
        print(f"  {key:14s} = {value}")
    print("=" * 60)

    # ---------- 1. 收集数据 ----------
    t0 = time.time()
    print(f"\n收集 {args.episodes} 局数据（policy={args.policy}）...")
    episodes = collect_episodes(
        args.episodes,
        seed=args.seed,
        policy=args.policy,
        max_steps=args.max_steps,
        grid_size=args.grid_size,
        enable_abstain=args.enable_abstain,
        abstain_prob=args.abstain_prob,
        abstain_penalty=args.abstain_penalty,
        random_abstain_prob=args.random_abstain_prob,
        ask_costs=ask_costs,
    )
    print(f"收集完成，用时 {time.time() - t0:.1f} 秒，共 {len(episodes)} 局。")

    # 90% 训练 / 10% 验证
    n_val = max(1, int(0.1 * len(episodes)))
    train_episodes = episodes[:-n_val]
    val_episodes = episodes[-n_val:]
    train_ds = EpisodeDataset(train_episodes, seq_len=args.seq_len, seed=args.seed + 1)
    val_ds = EpisodeDataset(val_episodes, seq_len=args.seq_len, seed=args.seed + 2)

    # ---------- 2. 建模型 ----------
    model = WorldModel(
        obs_dim=OBS_DIM,
        msg_dim=MSG_DIM,
        n_actions=n_actions,
        use_decoder=not args.no_decoder,
        use_ask_head=args.learned_ask,
        ask_cost_scale=args.ask_cost_scale if args.learned_ask else 1.0,
    )
    print(f"\n模型参数量（可训练）：{count_parameters(model):,}")
    print(f"是否使用观测解码器：{not args.no_decoder}")
    print()

    # ---------- 3. 训练 ----------
    t_train = time.time()
    history = train_world_model(
        model,
        train_ds,
        steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        log_every=args.log_every,
        ask_w=args.learned_ask_weight,
    )
    train_seconds = time.time() - t_train

    # ---------- 4. 评估 ----------
    print("\n在验证集上评估...")
    eval_metrics = evaluate_world_model(
        model,
        val_ds,
        n_batches=args.eval_batches,
        batch_size=args.batch,
    )

    print("\n验证集指标：")
    for key in [
        "reward_mae",
        "value_mae",
        "belief_acc",
        "brier",
        "ask_acc",
        "ask_brier",
        "jepa_cos",
        "kl",
        "kl_raw",
        "active_dims",
        "post_std",
        "prior_std",
        "z_std",
        "probe_acc",
        "majority_baseline",
        "effective_rank",
    ]:
        if key in eval_metrics:
            print(f"  {key:18s} = {eval_metrics[key]:.4f}")

    # ---------- 5. 保存 ----------
    config = {
        "obs_dim": OBS_DIM,
        "msg_dim": MSG_DIM,
        "n_actions": n_actions,
        "e_dim": 64,
        "h_dim": 96,
        "z_dim": 16,
        "use_decoder": not args.no_decoder,
        "use_ask_head": args.learned_ask,
        "ask_cost_scale": args.ask_cost_scale if args.learned_ask else 1.0,
    }
    env_config = {
        "grid_size": args.grid_size,
        "max_steps": args.max_steps,
        "enable_abstain": args.enable_abstain,
        "abstain_penalty": args.abstain_penalty,
        "ask_costs": ask_costs,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "env_config": env_config,
            "args": vars(args),
        },
        out_path,
    )
    print(f"\n已保存 checkpoint：{out_path.resolve()}")

    payload = {
        "args": vars(args),
        "config": config,
        "env_config": env_config,
        "num_parameters": count_parameters(model),
        "train_seconds": round(train_seconds, 1),
        "eval": eval_metrics,
        "history": history,
    }
    save_json(args.metrics_out, payload)
    print(f"已保存指标：{Path(args.metrics_out).resolve()}")

    print("\n下一步：")
    print("  python steps/03_plan.py --episodes 100 --horizon 8 --samples 32")
    print("  对比 --no-decoder 与默认版本的参数量、probe_acc、reward_mae。")


if __name__ == "__main__":
    main()
