"""
把 experiments/logs/ 里的 JSON 汇总成报告用的表格和图。

用法（项目根目录）：
    python experiments/summarize.py

输出：
    report/tables.md
    report/figures/h1_compare.png
    report/figures/h2_budget.png
    report/figures/h3_ask.png
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "experiments" / "logs"
REPORT = ROOT / "report"
FIGS = REPORT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]


def load(name: str) -> dict:
    path = LOGS / name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.{digits}f}"


def fmt_ms(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}"


# ---------- H1 ----------
def h1_table() -> list[str]:
    rows = []
    for kind in ["dec", "nodec"]:
        metrics = []
        for seed in SEEDS:
            name = (
                f"world_model_{kind}_metrics.json"
                if seed == 0
                else f"world_model_{kind}_s{seed}.json"
            )
            metrics.append(load(name))

        def ms(key):
            return mean_std([m["eval"][key] for m in metrics])

        params = metrics[0]["num_parameters"]
        train_s = mean_std([m["train_seconds"] for m in metrics])
        reward = ms("reward_mae")
        value = ms("value_mae")
        belief = ms("belief_acc")
        brier = ms("brier")
        jepa = ms("jepa_cos")
        probe = ms("probe_acc")
        majority = ms("majority_baseline")
        effrank = ms("effective_rank")

        rows.append(
            f"| {'有解码器' if kind == 'dec' else '无解码器'} | {params:,} | "
            f"{train_s[0]:.1f} ± {train_s[1]:.1f} | "
            f"{fmt(reward[0])} ± {fmt(reward[1])} | "
            f"{fmt(value[0])} ± {fmt(value[1])} | "
            f"{fmt(belief[0])} ± {fmt(belief[1])} | "
            f"{fmt(brier[0])} ± {fmt(brier[1])} | "
            f"{fmt(jepa[0])} ± {fmt(jepa[1])} | "
            f"{fmt(probe[0])} ± {fmt(probe[1])} | "
            f"{fmt(majority[0])} ± {fmt(majority[1])} | "
            f"{fmt(effrank[0], 2)} ± {fmt(effrank[1], 2)} |"
        )
    header = (
        "| 版本 | 参数量 | 训练秒 | reward MAE | value MAE | belief acc | Brier | JEPA cos | probe acc | 多数类 | 有效秩 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    return [header, *rows]


# ---------- H2 ----------
def h2_table() -> list[str]:
    configs = [(1, 8), (8, 8), (32, 8), (128, 8), (32, 4), (32, 12)]
    rows = []
    for n, h in configs:
        ms = []
        for seed in SEEDS:
            ms.append(load(f"h2_n{n}_h{h}_s{seed}.json"))
        success = mean_std([m["success_rate"] for m in ms])
        ret = mean_std([m["avg_return"] for m in ms])
        steps = mean_std([m["avg_steps"] for m in ms])
        latency = mean_std([m["avg_planner_ms_per_step"] for m in ms])
        rows.append(
            f"| N={n:<3d}, H={h:<2d} | "
            f"{fmt(success[0])} ± {fmt(success[1])} | "
            f"{fmt(ret[0])} ± {fmt(ret[1])} | "
            f"{fmt(steps[0], 1)} ± {fmt(steps[1], 1)} | "
            f"{fmt_ms(*latency)} |"
        )
    header = (
        "| 配置 | 成功率 | 平均回报 | 平均步数 | 单步规划 ms |\n"
        "|---|---:|---:|---:|---:|"
    )
    return [header, *rows]


# ---------- H3 ----------
H3_ROWS = [
    ("sf_never_low", "先读牌 + 从不问", "0.00"),
    ("sf_fixed_low", "先读牌 + 固定问", "-0.05"),
    ("sf_fixed_high", "先读牌 + 固定问", "-0.40"),
    ("sf_entropy08_low", "先读牌 + 熵门控 τ=0.8", "-0.05"),
    ("sf_entropy08_high", "先读牌 + 熵门控 τ=0.8", "-0.40"),
    ("sf_voi_low", "先读牌 + VOI 门控", "-0.05"),
    ("sf_voi_high", "先读牌 + VOI 门控", "-0.40"),
    ("nosf_fixed_low", "不先读牌 + 固定问", "-0.05"),
    ("nosf_voi_high", "不先读牌 + VOI 门控", "-0.40"),
]


def h3_table() -> list[str]:
    rows = []
    for tag, label, cost in H3_ROWS:
        ms = [load(f"h3_{tag}_s{seed}.json") for seed in SEEDS]
        success = mean_std([m["success_rate"] for m in ms])
        ret = mean_std([m["avg_return"] for m in ms])
        ask = mean_std([m["ask_rate"] for m in ms])
        ask_n = mean_std([m["avg_ask_count"] for m in ms])
        steps = mean_std([m["avg_steps"] for m in ms])
        rows.append(
            f"| {label} | {cost} | "
            f"{fmt(success[0])} ± {fmt(success[1])} | "
            f"{fmt(ret[0])} ± {fmt(ret[1])} | "
            f"{fmt(ask[0])} ± {fmt(ask[1])} | "
            f"{fmt(ask_n[0], 2)} ± {fmt(ask_n[1], 2)} | "
            f"{fmt(steps[0], 1)} ± {fmt(steps[1], 1)} |"
        )
    header = (
        "| 策略 | 询问成本 | 成功率 | 平均回报 | 询问率 | 平均询问次数 | 平均步数 |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    return [header, *rows]


# ---------- 图 ----------
def plot_h1():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = ["decoder", "decoder-free"]
    probes = []
    rewards = []
    params = []
    for kind in ["dec", "nodec"]:
        ms = []
        for seed in SEEDS:
            name = (
                f"world_model_{kind}_metrics.json"
                if seed == 0
                else f"world_model_{kind}_s{seed}.json"
            )
            ms.append(load(name))
        probes.append([m["eval"]["probe_acc"] for m in ms])
        rewards.append([m["eval"]["reward_mae"] for m in ms])
        params.append(ms[0]["num_parameters"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, data, title in zip(
        axes,
        [probes, rewards],
        ["Linear probe accuracy (goal)", "Reward prediction MAE"],
    ):
        means = [np.mean(x) for x in data]
        stds = [np.std(x, ddof=1) for x in data]
        ax.bar(kinds, means, yerr=stds, capsize=4, color=["#7aa6c2", "#e0a458"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0.75, 1.0)
    fig.suptitle(
        f"Decoder vs decoder-free (params: {params[0]:,} vs {params[1]:,})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "h1_compare.png", dpi=150)
    plt.close(fig)


def plot_h2():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [1, 8, 32, 128]
    success_n = []
    ret_n = []
    latency_n = []
    for n in ns:
        ms = [load(f"h2_n{n}_h8_s{seed}.json") for seed in SEEDS]
        success_n.append([m["success_rate"] for m in ms])
        ret_n.append([m["avg_return"] for m in ms])
        latency_n.append([m["avg_planner_ms_per_step"] for m in ms])

    hs = [4, 8, 12]
    success_h = []
    ret_h = []
    for h in hs:
        ms = [load(f"h2_n32_h{h}_s{seed}.json") for seed in SEEDS]
        success_h.append([m["success_rate"] for m in ms])
        ret_h.append([m["avg_return"] for m in ms])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].errorbar(
        ns, [np.mean(x) for x in success_n], yerr=[np.std(x, ddof=1) for x in success_n],
        marker="o", capsize=4,
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Samples N (log2)")
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("More thinking (H=8)")
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(
        hs, [np.mean(x) for x in success_h], yerr=[np.std(x, ddof=1) for x in success_h],
        marker="o", capsize=4, color="#e0a458",
    )
    axes[1].set_xlabel("Horizon H (N=32)")
    axes[1].set_ylabel("Success rate")
    axes[1].set_title("Longer imagination")
    axes[1].grid(alpha=0.3)

    axes[2].errorbar(
        ns, [np.mean(x) for x in latency_n], yerr=[np.std(x, ddof=1) for x in latency_n],
        marker="o", capsize=4, color="#6a994e",
    )
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Samples N (log2)")
    axes[2].set_ylabel("ms per decision")
    axes[2].set_title("Cost of thinking")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS / "h2_budget.png", dpi=150)
    plt.close(fig)


def plot_h3():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = [
        ("fixed", "sf_fixed_low", "sf_fixed_high"),
        ("entropy", "sf_entropy08_low", "sf_entropy08_high"),
        ("VOI", "sf_voi_low", "sf_voi_high"),
    ]
    low_means, low_stds = [], []
    high_means, high_stds = [], []
    for _, low_tag, high_tag in policies:
        low = [load(f"h3_{low_tag}_s{seed}.json")["avg_return"] for seed in SEEDS]
        high = [load(f"h3_{high_tag}_s{seed}.json")["avg_return"] for seed in SEEDS]
        low_means.append(np.mean(low))
        low_stds.append(np.std(low, ddof=1))
        high_means.append(np.mean(high))
        high_stds.append(np.std(high, ddof=1))

    x = np.arange(len(policies))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - width / 2, low_means, width, yerr=low_stds, capsize=4, label="ask cost 0.05")
    ax.bar(x + width / 2, high_means, width, yerr=high_stds, capsize=4, label="ask cost 0.40")
    never = [load(f"h3_sf_never_low_s{seed}.json")["avg_return"] for seed in SEEDS]
    ax.axhline(np.mean(never), linestyle="--", color="gray", label="never ask")
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in policies])
    ax.set_ylabel("Average return")
    ax.set_title("When is asking worth it? (sign-first)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "h3_ask.png", dpi=150)
    plt.close(fig)


def main():
    parts = [
        "# AskDreamer 实验表格（自动生成）\n",
        "## H1：无解码器 vs 有解码器（3 seeds，均值 ± 样本标准差）\n",
        *h1_table(),
        "\n## H2：思考预算（3 seeds，30 局/seed）\n",
        *h2_table(),
        "\n## H3：询问策略（3 seeds，50 局/seed，先读告示牌）\n",
        *h3_table(),
        "\n> 注：`never` 不产生询问成本，因此高/低询问成本下结果相同。\n",
        "> 注：世界模型在 `ask_cost=-0.05` 的数据上训练；H3 高成本（−0.40）评估存在奖励/价值尺度偏移，详见 `report.md` 第 8.2 节。\n",
        "> 注：`sign_first` 是规则式探索先验，不是学习得到的策略。\n",
    ]
    text = "\n".join(parts)
    (REPORT / "tables.md").write_text(text, encoding="utf-8")
    print(text)

    plot_h1()
    plot_h2()
    plot_h3()
    print(f"\n已写入：{REPORT / 'tables.md'}")
    print(f"已写入图：{FIGS}")


if __name__ == "__main__":
    main()
