"""
任务 1：分布外（OOD）评估。

同一个 checkpoint，分别在 train-grid 和 test-grid 上跑同一套规划/询问流程，
比较成功率、回报、步数、信念校准（accuracy / Brier）和询问率。

示例：
    python experiments/eval_ood.py \
      --checkpoint experiments/checkpoints/world_model_nodec.pt \
      --train-grid 7 --test-grid 10 \
      --episodes 50 --samples 32 --horizon 8 --ask never --seed 0

输出：
    experiments/logs/ood_7to10_seed0.json
    report/figures/ood_compare.png

启用 ABSTAIN（6 动作 checkpoint）时，额外报告 abstain_rate / coverage /
selective_success，默认输出到 ood_*_abstain_<gate>.json 和
report/figures/ood_abstain_compare.png，不会覆盖上面的任务 1 结果。

注意：这是“用 7x7 训练、直接搬到 10x10”的分布外测试，不会重新训练，
也不会覆盖任何已有 checkpoint 或报告文件。
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from askdreamer.controller import (
    belief_probs,
    make_abstain_policy,
    make_ask_policy,
)
from askdreamer.env import (
    ABSTAIN_ACTION,
    ASK_ACTION,
    START_POS,
    HiddenGoalGrid,
)
from askdreamer.model import WorldModel
from askdreamer.planner import RandomShootingPlanner
from askdreamer.train import save_json, seed_everything

# 参考色板（light surface），已通过 dataviz 调色板校验。
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
COLOR_TRAIN = "#2a78d6"
COLOR_TEST = "#eb6834"

METRIC_LABELS = {
    "success_rate": "成功率",
    "avg_return": "平均回报",
    "avg_steps": "平均步数",
    "belief_acc": "信念准确率",
    "brier": "Brier（越低越好）",
    "belief_entropy": "信念熵",
    "ask_rate": "询问率",
    "avg_ask_count": "平均询问次数",
    "sign_rate": "读告示牌比例",
    "timeout_rate": "超时比例",
    "abstain_rate": "弃权率",
    "coverage": "覆盖率（未弃权）",
    "selective_success": "覆盖率内成功率",
}

FIGURE_PANELS = [
    ("success_rate", "Success rate", True),
    ("avg_return", "Average return", False),
    ("avg_steps", "Average steps", False),
    ("belief_acc", "Belief accuracy", True),
    ("brier", "Brier score (lower is better)", False),
    ("ask_rate", "Ask rate", True),
]


def parse_args():
    p = argparse.ArgumentParser(description="7x7 -> 10x10 分布外评估")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/world_model_nodec.pt",
        help="默认使用报告里主用的 7x7 无解码器权重",
    )
    p.add_argument("--train-grid", type=int, default=7, help="训练时的网格边长")
    p.add_argument("--test-grid", type=int, default=10, help="测试时的网格边长")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--samples", type=int, default=32, help="规划器采样数")
    p.add_argument("--horizon", type=int, default=8, help="规划视野")
    p.add_argument(
        "--ask",
        type=str,
        default="never",
        choices=["never", "fixed", "entropy", "voi"],
        help="默认不问，让信念只能来自观测/历史；想看询问行为时用 entropy/voi",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--ask-after", type=int, default=0)
    p.add_argument("--ask-cost", type=float, default=-0.05)
    p.add_argument("--wrong-penalty", type=float, default=-0.3)
    p.add_argument(
        "--enable-abstain",
        action="store_true",
        help="启用第 6 个动作 ABSTAIN（需要 6 动作 checkpoint）",
    )
    p.add_argument(
        "--abstain",
        type=str,
        default="never",
        choices=["never", "entropy", "voi"],
        help="弃权门控；仅 --enable-abstain 时生效",
    )
    p.add_argument("--abstain-threshold", type=float, default=0.5)
    p.add_argument("--abstain-after", type=int, default=0)
    p.add_argument("--abstain-margin", type=float, default=0.0)
    p.add_argument("--abstain-penalty", type=float, default=-0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--no-value", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--figure",
        type=str,
        default=None,
        help="默认 ood_compare.png；启用 ABSTAIN 时默认 ood_abstain_compare.png",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_world_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = WorldModel(**ckpt["config"])
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"警告：checkpoint 缺少 {len(missing)} 个参数（旧版本？value_head 会随机初始化）")
    if unexpected:
        print(f"警告：checkpoint 多出 {len(unexpected)} 个参数（已忽略）")
    model.eval()
    return model, ckpt.get("config", {})


@torch.no_grad()
def eval_condition(model: WorldModel, grid_size: int, args, label: str) -> dict:
    """在指定网格上跑 args.episodes 局，返回汇总指标和逐局记录。"""
    env = HiddenGoalGrid(
        seed=args.seed,
        max_steps=args.max_steps,
        ask_cost=args.ask_cost,
        wrong_penalty=args.wrong_penalty,
        grid_size=grid_size,
        start_pos=START_POS,
        enable_abstain=args.enable_abstain,
        abstain_penalty=args.abstain_penalty,
    )
    if getattr(model, "n_actions", env.n_actions) != env.n_actions:
        raise SystemExit(
            f"checkpoint 动作数（{model.n_actions}）与当前环境（{env.n_actions}）不一致；"
            "请检查是否需要 --enable-abstain。"
        )
    planner = RandomShootingPlanner(
        model,
        horizon=args.horizon,
        samples=args.samples,
        gamma=args.gamma,
        use_value=not args.no_value,
    )
    ask_policy = make_ask_policy(args.ask, args.threshold, args.ask_after)
    abstain_policy = (
        make_abstain_policy(
            args.abstain,
            args.abstain_threshold,
            args.abstain_after,
            args.abstain_margin,
        )
        if args.enable_abstain
        else None
    )

    records = []
    belief_n = 0
    belief_correct = 0
    brier_sum = 0.0
    entropy_sum = 0.0
    t_start = time.time()

    print(f"\n[{label}] grid={grid_size}x{grid_size}, episodes={args.episodes}")

    for ep in range(args.episodes):
        episode_seed = args.seed * 100_000 + ep
        obs, msg = env.reset(seed=episode_seed)

        h = torch.zeros(1, model.h_dim)
        z = torch.zeros(1, model.z_dim)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        msg_t = torch.tensor(msg, dtype=torch.float32).unsqueeze(0)
        h, z, _, _ = model.observe_step(h, z, None, msg_t, obs_t)

        total_reward = 0.0
        planner_ms = 0.0
        planner_calls = 0
        done = False

        while not done:
            # 决策状态下的信念：动作选择之前采样，和训练语义一致。
            probs = belief_probs(model, h, z)[0]
            goal = env.goal
            belief_n += 1
            belief_correct += int(int(probs.argmax().item()) == goal)
            onehot = torch.zeros_like(probs)
            onehot[goal] = 1.0
            brier_sum += float(((probs - onehot) ** 2).sum().item())
            entropy_sum += float(-(probs * torch.log(probs + 1e-8)).sum().item())

            # 与 steps/03_plan.py 相同的动作选择顺序。
            action = None
            if hasattr(ask_policy, "override_action"):
                action = ask_policy.override_action(env)
            if action is None:
                if ask_policy.should_ask(model, h, z, env):
                    action = ASK_ACTION
                elif abstain_policy is not None and abstain_policy.should_abstain(
                    model, h, z, env
                ):
                    action = ABSTAIN_ACTION
                else:
                    t0 = time.time()
                    action, _ = planner.act(h, z, msg_t)
                    planner_ms += (time.time() - t0) * 1000.0
                    planner_calls += 1

            obs, msg, reward, done, info = env.step(action)
            total_reward += reward

            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            msg_t = torch.tensor(msg, dtype=torch.float32).unsqueeze(0)
            h, z, _, _ = model.observe_step(
                h, z, torch.tensor([action]), msg_t, obs_t
            )

        record = {
            "episode": ep,
            "seed": episode_seed,
            "return": float(total_reward),
            "steps": int(env.steps),
            "success": bool(info["success"]),
            "ask_count": int(info["ask_count"]),
            "sign_read": bool(info["sign_read"]),
            "timeout": bool(info["timeout"]),
            "abstained": bool(info["abstained"]),
            "planner_ms": planner_ms / max(1, planner_calls),
        }
        records.append(record)
        if args.verbose:
            print(
                f"  ep {ep:3d}  return={record['return']:+.2f}  "
                f"steps={record['steps']:2d}  success={record['success']}  "
                f"ask={record['ask_count']}"
            )

    n = len(records)

    def mean(key):
        return float(np.mean([r[key] for r in records])) if n else 0.0

    n_covered = sum(1 for r in records if not r["abstained"])
    coverage = n_covered / n if n else 0.0
    selective_success = (
        sum(1 for r in records if r["success"] and not r["abstained"]) / n_covered
        if n_covered > 0
        else 0.0
    )

    summary = {
        "grid_size": grid_size,
        "episodes": n,
        "success_rate": mean("success"),
        "avg_return": mean("return"),
        "avg_steps": mean("steps"),
        "belief_acc": belief_correct / max(1, belief_n),
        "brier": brier_sum / max(1, belief_n),
        "belief_entropy": entropy_sum / max(1, belief_n),
        "ask_rate": float(np.mean([r["ask_count"] > 0 for r in records])) if n else 0.0,
        "avg_ask_count": mean("ask_count"),
        "sign_rate": mean("sign_read"),
        "timeout_rate": mean("timeout"),
        "abstain_rate": mean("abstained"),
        "coverage": coverage,
        "selective_success": selective_success,
        "n_belief_samples": belief_n,
        "avg_planner_ms_per_step": mean("planner_ms"),
        "total_seconds": round(time.time() - t_start, 2),
        "per_episode": records,
    }
    abstain_note = (
        f"  abstain_rate={summary['abstain_rate']:.3f}  "
        f"coverage={summary['coverage']:.3f}"
        if args.enable_abstain
        else ""
    )
    print(
        f"[{label}] success={summary['success_rate']:.3f}  "
        f"return={summary['avg_return']:+.3f}  steps={summary['avg_steps']:.1f}  "
        f"belief_acc={summary['belief_acc']:.3f}  brier={summary['brier']:.3f}  "
        f"ask_rate={summary['ask_rate']:.3f}" + abstain_note
    )
    return summary


def summarize_for_json(summary: dict) -> dict:
    """去掉逐局明细，便于和已有日志格式对齐。"""
    return {k: v for k, v in summary.items() if k != "per_episode"}


def print_comparison(train: dict, test: dict, train_grid: int, test_grid: int):
    print("\n" + "=" * 78)
    print(f"{'指标':<20} {'train ' + str(train_grid) + 'x' + str(train_grid):>16} "
          f"{'test ' + str(test_grid) + 'x' + str(test_grid):>16} {'delta':>14}")
    print("-" * 78)
    for key in [
        "success_rate",
        "avg_return",
        "avg_steps",
        "belief_acc",
        "brier",
        "belief_entropy",
        "ask_rate",
        "avg_ask_count",
        "sign_rate",
        "timeout_rate",
    ]:
        a, b = float(train[key]), float(test[key])
        print(
            f"{METRIC_LABELS.get(key, key):<20} {a:>16.4f} {b:>16.4f} "
            f"{b - a:>+14.4f}"
        )
    if train.get("abstain_rate", 0.0) > 0 or test.get("abstain_rate", 0.0) > 0:
        for key in ["abstain_rate", "coverage", "selective_success"]:
            a, b = float(train[key]), float(test[key])
            print(
                f"{METRIC_LABELS.get(key, key):<20} {a:>16.4f} {b:>16.4f} "
                f"{b - a:>+14.4f}"
            )
    print("=" * 78)
    print(f"信念采样步数：train={train['n_belief_samples']}, test={test['n_belief_samples']}")
    print("参考值：均匀信念下 belief_acc=1/3≈0.333，Brier=2/3≈0.667")


def make_figure(train: dict, test: dict, train_grid: int, test_grid: int,
                args, path: str):
    """2x3 小多图：每个指标两根柱（train vs test），只在 test 柱顶直接标值。"""
    fig, axes = plt.subplots(
        2, 3, figsize=(12.8, 6.6), facecolor=SURFACE, constrained_layout=True
    )
    xlabels = [f"{train_grid}x{train_grid}", f"{test_grid}x{test_grid}"]

    panels = list(FIGURE_PANELS)
    if args.enable_abstain:
        # 启用 ABSTAIN 时，最后一格改为弃权率；询问率仍保留在 JSON 和控制台。
        panels[-1] = ("abstain_rate", "Abstain rate", True)

    for ax, (key, title, is_rate) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        values = [float(train[key]), float(test[key])]
        ax.bar(
            [0, 1],
            values,
            width=0.46,
            color=[COLOR_TRAIN, COLOR_TEST],
            edgecolor="none",
            zorder=3,
        )

        ax.set_title(title, fontsize=11, color=INK, pad=10)
        ax.set_xticks([0, 1], xlabels, fontsize=9, color=INK_SECONDARY)
        ax.tick_params(axis="y", labelsize=8, colors=INK_MUTED, length=0)
        ax.tick_params(axis="x", length=0)

        ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.spines["bottom"].set_linewidth(1)

        # 零线 / 参考线
        if min(values) < 0:
            ax.axhline(0, color=BASELINE, linewidth=1, zorder=2)
        if key == "belief_acc":
            ax.axhline(1 / 3, color=INK_MUTED, linewidth=1, zorder=2)
            ax.annotate(
                "chance 1/3", (1.42, 1 / 3), fontsize=8, color=INK_MUTED,
                ha="right", va="bottom",
            )
        elif key == "brier":
            ax.axhline(2 / 3, color=INK_MUTED, linewidth=1, zorder=2)
            ax.annotate(
                "chance 2/3", (1.42, 2 / 3), fontsize=8, color=INK_MUTED,
                ha="right", va="bottom",
            )

        # 纵轴范围
        if is_rate:
            lo = min(0.0, min(values) - 0.05)
            ax.set_ylim(lo, 1.06)
        elif key in ("brier", "belief_entropy"):
            ax.set_ylim(0, max(0.8, max(values) * 1.25))
        elif key == "avg_steps":
            ax.set_ylim(0, max(args.max_steps, max(values)) * 1.15)
        else:
            span = max(abs(v) for v in values) or 1.0
            ax.set_ylim(min(0.0, min(values) - 0.1 * span),
                        max(0.0, max(values) + 0.15 * span))

        # 只给 test 柱直接标值（选择性标注）
        v = values[1]
        ax.annotate(
            f"{v:.3f}",
            (1, v),
            xytext=(0, 4 if v >= 0 else -11),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
        if key == "ask_rate" and max(values) == 0:
            ax.annotate(
                f"no asking (ask={args.ask})",
                (0.5, 0.12),
                ha="center",
                fontsize=8.5,
                color=INK_MUTED,
            )
        if key == "abstain_rate" and max(values) == 0:
            ax.annotate(
                f"no abstaining (gate={args.abstain})",
                (0.5, 0.12),
                ha="center",
                fontsize=8.5,
                color=INK_MUTED,
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_TRAIN, edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_TEST, edgecolor="none"),
    ]
    fig.legend(
        handles,
        [f"train {train_grid}x{train_grid}", f"test {test_grid}x{test_grid}"],
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        bbox_to_anchor=(0.995, 1.055),
    )
    fig.suptitle(
        f"OOD evaluation: checkpoint trained on {train_grid}x{train_grid}, "
        f"evaluated on {train_grid}x{train_grid} vs {test_grid}x{test_grid}",
        fontsize=13,
        color=INK,
        x=0.01,
        ha="left",
        y=1.04,
    )
    fig.text(
        0.01,
        1.0,
        f"episodes={args.episodes}  samples={args.samples}  horizon={args.horizon}  "
        f"ask={args.ask}  seed={args.seed}  |  "
        f"{Path(args.checkpoint).name}"
        + (f"  abstain={args.abstain}" if args.enable_abstain else ""),
        fontsize=9,
        color=INK_MUTED,
        ha="left",
        va="top",
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"\n已保存对比图：{out.resolve()}")


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(3)

    print("=" * 60)
    print("OOD 评估配置：")
    for key, value in vars(args).items():
        print(f"  {key:16s} = {value}")
    print("=" * 60)

    model, ckpt_config = load_world_model(args.checkpoint)

    train_summary = eval_condition(model, args.train_grid, args, "train")
    test_summary = eval_condition(model, args.test_grid, args, "test")

    print_comparison(train_summary, test_summary, args.train_grid, args.test_grid)

    comparison = {}
    comparison_keys = [
        "success_rate",
        "avg_return",
        "avg_steps",
        "belief_acc",
        "brier",
        "belief_entropy",
        "ask_rate",
        "avg_ask_count",
        "sign_rate",
        "timeout_rate",
        "abstain_rate",
        "coverage",
        "selective_success",
    ]
    for key in comparison_keys:
        a = float(train_summary[key])
        b = float(test_summary[key])
        comparison[key] = {"train": a, "test": b, "delta": b - a}

    out_path = Path(
        args.out
        or (
            f"experiments/logs/ood_{args.train_grid}to{args.test_grid}"
            f"_seed{args.seed}"
            + (f"_abstain_{args.abstain}" if args.enable_abstain else "")
            + ".json"
        )
    )
    payload = {
        "args": vars(args),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_config": ckpt_config,
        "train_grid": args.train_grid,
        "test_grid": args.test_grid,
        "train": summarize_for_json(train_summary),
        "test": summarize_for_json(test_summary),
        "comparison": comparison,
        "per_episode": {
            "train": train_summary["per_episode"],
            "test": test_summary["per_episode"],
        },
    }
    save_json(out_path, payload)
    print(f"已保存指标：{out_path.resolve()}")

    figure_path = args.figure or (
        "report/figures/ood_abstain_compare.png"
        if args.enable_abstain
        else "report/figures/ood_compare.png"
    )
    make_figure(
        train_summary,
        test_summary,
        args.train_grid,
        args.test_grid,
        args,
        figure_path,
    )


if __name__ == "__main__":
    main()
