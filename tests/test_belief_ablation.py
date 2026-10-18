"""任务 4 的回归测试：不使用 goal 标签训练信念头。

运行：
    python -m pytest tests/test_belief_ablation.py -q
"""

from __future__ import annotations

import pytest
import torch

from askdreamer.controller import (
    EntropyAbstain,
    EntropyGate,
    VOIAbstain,
    VOIGate,
)
from askdreamer.data import EpisodeDataset, collect_episodes
from askdreamer.env import HiddenGoalGrid
from askdreamer.model import WorldModel


def test_default_model_keeps_belief_supervision():
    model = WorldModel(use_decoder=False)
    assert model.use_belief_supervision is True
    assert model.belief_head is not None


def test_ablation_skips_belief_loss():
    episodes = collect_episodes(6, seed=0, policy="scripted", max_steps=8)
    batch = EpisodeDataset(episodes, seq_len=6, seed=1).sample_batch(4)

    model = WorldModel(use_decoder=False, use_belief_supervision=False)
    loss, metrics = model.compute_losses(batch)

    assert torch.isfinite(loss)
    assert float(metrics["loss_belief"]) == 0.0
    # 监控指标仍会计算，便于确认该头没有学到 goal 信息
    assert 0.0 <= float(metrics["belief_acc"]) <= 1.0


def test_belief_gates_require_supervision():
    model = WorldModel(use_decoder=False, use_belief_supervision=False)
    env = HiddenGoalGrid(seed=0)
    env.reset(seed=0)

    h = torch.zeros(1, model.h_dim)
    z = torch.zeros(1, model.z_dim)

    with pytest.raises(RuntimeError):
        EntropyGate().should_ask(model, h, z, env)
    with pytest.raises(RuntimeError):
        VOIGate().should_ask(model, h, z, env)

    env = HiddenGoalGrid(seed=0, enable_abstain=True)
    env.reset(seed=0)
    env.steps = 5
    env.agent_pos = (1, 5)  # 门 (1,6) 旁边
    with pytest.raises(RuntimeError):
        EntropyAbstain().should_abstain(model, h, z, env)
    with pytest.raises(RuntimeError):
        VOIAbstain().should_abstain(model, h, z, env)
