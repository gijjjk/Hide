"""任务 3 的回归测试：学习型询问门控（ask_head）。

运行：
    python -m pytest tests/test_learned_ask.py -q
"""

from __future__ import annotations

import pytest
import torch

from askdreamer.controller import LearnedAsk, make_ask_policy
from askdreamer.data import EpisodeDataset, ask_label, collect_episodes
from askdreamer.env import HiddenGoalGrid
from askdreamer.model import WorldModel


def test_ask_label_depends_on_cost_and_remaining_doors():
    low = HiddenGoalGrid(seed=0, ask_cost=-0.05)
    low.reset(seed=0)
    assert ask_label(low) == 1.0

    high = HiddenGoalGrid(seed=0, ask_cost=-0.4)
    high.reset(seed=0)
    assert ask_label(high) == 0.0

    # 告示牌排除一扇门后只剩 2 扇：期望试错成本约 0.155
    medium = HiddenGoalGrid(seed=0, ask_cost=-0.2)
    medium.reset(seed=0)
    medium.sign_read = True
    medium.msg[0] = 1.0
    assert ask_label(medium) == 0.0

    # 已经问过就不再需要询问
    low.ask_count = 1
    assert ask_label(low) == 0.0


def test_collect_episodes_records_cost_and_label():
    episodes = collect_episodes(
        6, seed=0, policy="scripted", max_steps=10, ask_costs=[-0.05, -0.4]
    )
    assert len(episodes) == 6
    for ep in episodes:
        assert ep["cost"].shape == ep["ask_label"].shape == ep["obs"].shape[:1]
        # 每局的成本应属于给定集合，且标签与成本一致
        valid_costs = set(
            round(float(c), 3) for c in ep["cost"][ep["valid"] > 0]
        )
        assert len(valid_costs) == 1
        assert valid_costs <= {-0.05, -0.4}
        assert set(float(v) for v in ep["ask_label"]).issubset({0.0, 1.0})


def test_ask_head_loss_and_metrics():
    episodes = collect_episodes(
        8, seed=1, policy="scripted", max_steps=10, ask_costs=[-0.05, -0.4]
    )
    ds = EpisodeDataset(episodes, seq_len=8, seed=2)
    batch = ds.sample_batch(4)

    model = WorldModel(use_ask_head=True, use_decoder=False)
    loss, metrics = model.compute_losses(batch)

    assert torch.isfinite(loss)
    assert "loss_ask" in metrics
    assert 0.0 <= float(metrics["ask_acc"]) <= 1.0
    assert 0.0 <= float(metrics["ask_brier"]) <= 1.0

    h = torch.zeros(1, model.h_dim)
    z = torch.zeros(1, model.z_dim)
    prob = model.ask_prob(h, z, torch.tensor([0.05]))
    assert 0.0 <= float(prob.item()) <= 1.0


def test_model_without_ask_head_rejects_learned_policy():
    model = WorldModel(use_ask_head=False, use_decoder=False)
    env = HiddenGoalGrid(seed=0)
    env.reset(seed=0)
    gate = LearnedAsk()
    with pytest.raises(RuntimeError):
        gate.should_ask(model, torch.zeros(1, model.h_dim), torch.zeros(1, model.z_dim), env)


class _StubModel:
    ask_head = object()

    def __init__(self, prob: float):
        self.prob = prob

    def ask_prob(self, h, z, cost):
        return torch.tensor([self.prob])


def test_learned_gate_threshold_and_ask_count():
    env = HiddenGoalGrid(seed=0)
    env.reset(seed=0)
    env.steps = 3

    h = torch.zeros(1, 96)
    z = torch.zeros(1, 16)
    assert LearnedAsk(threshold=0.5).should_ask(_StubModel(0.9), h, z, env) is True
    assert LearnedAsk(threshold=0.95).should_ask(_StubModel(0.9), h, z, env) is False

    env.ask_count = 1
    assert LearnedAsk(threshold=0.5).should_ask(_StubModel(0.9), h, z, env) is False


def test_ask_cost_scale_changes_cost_sensitivity():
    model = WorldModel(use_ask_head=True, use_decoder=False, ask_cost_scale=20.0)
    head = torch.nn.Linear(model.h_dim + model.z_dim + 1, 1)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[0, -1] = -1.0  # 只让成本影响输出
        head.bias.zero_()
    model.ask_head = head

    h = torch.zeros(1, model.h_dim)
    z = torch.zeros(1, model.z_dim)
    assert torch.allclose(
        model.ask_prob(h, z, torch.tensor([0.05])),
        torch.sigmoid(torch.tensor([-1.0])),
        atol=1e-6,
    )
    assert torch.allclose(
        model.ask_prob(h, z, torch.tensor([0.4])),
        torch.sigmoid(torch.tensor([-8.0])),
        atol=1e-6,
    )


def test_make_ask_policy_learned():
    assert isinstance(make_ask_policy("learned"), LearnedAsk)
