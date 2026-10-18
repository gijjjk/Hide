"""
小 batch 过拟合测试：确认世界模型真的能学，而不是 shape 对了但梯度没生效。

运行方式（在 D:\\hide 下）：
    python tests/test_overfit.py

做法：收集少量脚本策略数据，固定一个 batch，训练 150 步；
要求最后 10 步的平均 reward MAE 明显低于最初 10 步。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import torch

from askdreamer.data import EpisodeDataset, collect_episodes
from askdreamer.model import WorldModel
from askdreamer.train import seed_everything

torch.set_num_threads(3)


def main():
    seed_everything(0)
    torch.manual_seed(0)

    episodes = collect_episodes(40, seed=0, policy="scripted")
    dataset = EpisodeDataset(episodes, seq_len=8, seed=1)
    batch = dataset.sample_batch(16)

    model = WorldModel(use_decoder=False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    first_mae = []
    last_mae = []
    n_steps = 150

    for step in range(1, n_steps + 1):
        loss, metrics = model.compute_losses(batch)
        assert torch.isfinite(loss), f"第 {step} 步 loss 不是有限值：{loss}"

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.update_target(decay=0.99)

        mae = float(metrics["reward_mae"])
        if step <= 10:
            first_mae.append(mae)
        if step > n_steps - 10:
            last_mae.append(mae)

        if step % 25 == 0 or step == 1:
            print(
                f"step {step:3d} | loss {loss.item():.4f} | "
                f"reward MAE {mae:.4f} | belief acc {float(metrics['belief_acc']):.3f}"
            )

    first = sum(first_mae) / len(first_mae)
    last = sum(last_mae) / len(last_mae)
    print(f"\n最初 10 步平均 reward MAE = {first:.4f}")
    print(f"最后 10 步平均 reward MAE = {last:.4f}")

    # 固定 batch + 150 步，reward 头应该能明显拟合；阈值留一点余量防随机波动
    assert last < first * 0.75, (first, last)
    print("✓ 过拟合测试通过：模型在同一 batch 上确实学到了东西。")


if __name__ == "__main__":
    main()
