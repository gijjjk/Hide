"""
世界模型的训练与评估工具。

训练分两段：
1. `train_world_model`：只训练世界模型（编码器 + RSSM + 各预测头）；
2. `evaluate_world_model`：在验证 episode 上算奖励误差、信念准确率/校准、JEPA 相似度、探针准确率。
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import EpisodeDataset
from .model import WorldModel


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU 上尽量可复现；如果有算子不支持确定性，只警告不崩溃
    torch.use_deterministic_algorithms(True, warn_only=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_world_model(
    model: WorldModel,
    dataset: EpisodeDataset,
    steps: int = 3000,
    batch_size: int = 32,
    lr: float = 3e-4,
    log_every: int = 100,
    ema: float = 0.99,
    device: str = "cpu",
) -> list[dict]:
    """返回训练日志列表，每个元素是一组标量指标。"""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    history = []
    t0 = time.time()

    # 打印一个 batch 的 shape
    first_batch = dataset.sample_batch(batch_size)
    print("一个 batch 的 shape：")
    for key, value in first_batch.items():
        print(f"  {key:10s} {tuple(value.shape)}")
    print()

    for step in range(1, steps + 1):
        # 每次重新采样；first_batch 只是为了打印 shape，不参与训练
        batch = first_batch if step == 1 else dataset.sample_batch(batch_size)
        batch = {k: v.to(device) for k, v in batch.items()}

        loss, metrics = model.compute_losses(batch)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"第 {step} 步 loss 变成 NaN/Inf。建议：降低 lr、检查 logvar clamp、先用 --decoder。"
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.update_target(decay=ema)

        if step % log_every == 0 or step == 1:
            row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
            row["elapsed_sec"] = round(time.time() - t0, 1)
            history.append(row)
            print(
                f"step {step:5d} | loss {row['loss']:.4f} | "
                f"reward_mae {row['reward_mae']:.4f} | "
                f"belief_acc {row['belief_acc']:.3f} | "
                f"v_mae {row['value_mae']:.3f} | "
                f"brier {row['brier']:.3f} | "
                f"jepa_cos {row['jepa_cos']:.3f} | "
                f"kl {row['kl']:.2f} | kl_raw {row['kl_raw']:.2f} | "
                f"active {row['active_dims']:.1f} | z_std {row['z_std']:.3f}"
            )

    print(f"\n训练完成，用时 {time.time() - t0:.1f} 秒。")
    return history


def evaluate_world_model(
    model: WorldModel,
    dataset: EpisodeDataset,
    n_batches: int = 10,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    """在世界模型的验证数据上算指标。"""
    model.eval()
    accum = {}
    count = 0

    with torch.no_grad():
        for _ in range(n_batches):
            batch = dataset.sample_batch(batch_size)
            batch = {k: v.to(device) for k, v in batch.items()}
            _, metrics = model.compute_losses(batch)
            for key, value in metrics.items():
                accum[key] = accum.get(key, 0.0) + float(value)
            count += 1

    result = {key: value / max(1, count) for key, value in accum.items()}

    # 附带潜变量质量指标（probe 需要梯度，所以放在 no_grad 外面）
    z, y = collect_latents_and_labels(
        model, dataset, n_batches=n_batches, batch_size=batch_size, device=device
    )
    result["probe_acc"] = linear_probe(z, y, z_dim=model.z_dim)
    result["majority_baseline"] = majority_baseline(y)
    result["effective_rank"] = effective_rank(z)
    return result


@torch.no_grad()
def collect_latents_and_labels(
    model: WorldModel,
    dataset: EpisodeDataset,
    n_batches: int = 10,
    batch_size: int = 32,
    device: str = "cpu",
):
    """收集 posterior 均值 z 和对应的隐藏门标签，用于线性探针。"""
    model.eval()
    zs = []
    ys = []
    for _ in range(n_batches):
        batch = dataset.sample_batch(batch_size)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.encode_sequence(batch["obs"], batch["msg"], batch["act"], sample_z=False)
        z = out["z"]                      # (B,L,z_dim)
        mask = batch["valid"].bool()      # (B,L)
        zs.append(z[mask].cpu())
        ys.append(batch["goal"][mask].cpu())

    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


def linear_probe(z: torch.Tensor, y: torch.Tensor, z_dim: int, seed: int = 0) -> float:
    """在冻结的 z 上训练一个线性分类器，衡量“隐藏门是否线性可读”。"""
    if z.shape[0] < 10:
        return float("nan")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(z.shape[0], generator=g)
    z, y = z[perm], y[perm]

    split = max(1, int(z.shape[0] * 0.8))
    z_train, y_train = z[:split], y[:split]
    z_val, y_val = z[split:], y[split:]
    if z_val.shape[0] == 0:
        z_val, y_val = z_train, y_train

    probe = nn.Linear(z_dim, 3)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)

    batch_size = min(256, z_train.shape[0])
    for _ in range(300):
        idx = torch.randint(0, z_train.shape[0], (batch_size,), generator=g)
        logits = probe(z_train[idx])
        loss = F.cross_entropy(logits, y_train[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = probe(z_val).argmax(dim=-1)
        acc = (pred == y_val).float().mean().item()
    return acc


def majority_baseline(y: torch.Tensor) -> float:
    """如果永远猜出现最多的类别，准确率是多少。"""
    if y.numel() == 0:
        return float("nan")
    counts = torch.bincount(y, minlength=3).float()
    return (counts.max() / counts.sum()).item()


@torch.no_grad()
def effective_rank(z: torch.Tensor, max_samples: int = 2000) -> float:
    """有效秩：越接近 z_dim，说明潜变量用了越多方向；接近 1 表示坍塌。"""
    if z.shape[0] < 2:
        return float("nan")
    if z.shape[0] > max_samples:
        z = z[:max_samples]
    z = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z)
    p = s / (s.sum() + 1e-9)
    entropy = -(p * torch.log(p + 1e-9)).sum()
    return math.exp(float(entropy))


def save_json(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
