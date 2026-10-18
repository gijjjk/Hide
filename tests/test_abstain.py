"""任务 2 的回归测试：ABSTAIN 动作与弃权门控。

运行：
    python -m pytest tests/test_abstain.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import askdreamer.controller as controller
from askdreamer.controller import (
    EntropyAbstain,
    NeverAbstain,
    VOIAbstain,
    _can_abstain,
    make_abstain_policy,
)
from askdreamer.data import ScriptedPolicy, collect_episodes
from askdreamer.env import (
    ABSTAIN_ACTION,
    N_ACTIONS,
    N_ACTIONS_WITH_ABSTAIN,
    HiddenGoalGrid,
)


def _adjacent_env(penalty: float = -0.1) -> HiddenGoalGrid:
    """把智能体放到第一扇门左边一格（信息不足、尚未询问）。"""
    env = HiddenGoalGrid(
        seed=0, enable_abstain=True, abstain_penalty=penalty
    )
    env.reset(seed=0)
    r, c = env.door_positions[0]
    env.agent_pos = (r, c - 1)
    env.ask_count = 0
    env.sign_read = False
    env.steps = 1
    return env


def test_default_action_space_unchanged():
    env = HiddenGoalGrid(seed=0)
    assert env.n_actions == N_ACTIONS == 5

    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(ABSTAIN_ACTION)


def test_abstain_terminates_with_penalty():
    env = HiddenGoalGrid(seed=0, enable_abstain=True, abstain_penalty=-0.2)
    env.reset(seed=0)
    assert env.n_actions == N_ACTIONS_WITH_ABSTAIN == 6

    _, _, reward, done, info = env.step(ABSTAIN_ACTION)

    assert done is True
    assert info["abstained"] is True
    assert info["success"] is False
    assert info["timeout"] is False
    assert reward == pytest.approx(env.step_penalty + env.abstain_penalty)
    assert env.steps == 1
    assert env.abstained is True

    # 新的一局必须清空弃权状态
    env.reset(seed=1)
    assert env.abstained is False


def test_scripted_policy_abstains_when_uncertain_and_adjacent():
    env = _adjacent_env()
    policy = ScriptedPolicy(
        np.random.default_rng(0),
        ask_prob=0.0,
        sign_prob=0.0,
        abstain_prob=1.0,
    )
    policy.visit_sign = False

    assert policy.act(env) == ABSTAIN_ACTION


def test_abstain_requires_enable_and_unknown_door():
    disabled = HiddenGoalGrid(seed=0)
    disabled.reset(seed=0)
    assert _can_abstain(disabled) is False

    env = _adjacent_env()
    assert _can_abstain(env) is True

    # 已经问过：信息买到了，不应该再弃权
    env.ask_count = 1
    assert _can_abstain(env) is False

    # 离门太远：不应该弃权
    env.ask_count = 0
    env.agent_pos = env.start_pos
    assert _can_abstain(env) is False


def test_entropy_and_voi_gates(monkeypatch):
    env = _adjacent_env()
    probs = torch.tensor([[0.2, 0.4, 0.4]])  # 熵 ≈ 1.055 nat，p_max=0.4
    monkeypatch.setattr(controller, "belief_probs", lambda model, h, z: probs)

    assert NeverAbstain().should_abstain(None, None, None, env) is False

    assert EntropyAbstain(threshold=0.5).should_abstain(None, None, None, env) is True
    assert EntropyAbstain(threshold=1.2).should_abstain(None, None, None, env) is False

    # 期望试错成本 ≈ (1-0.4)/0.4 * 0.31 ≈ 0.465
    assert VOIAbstain().should_abstain(None, None, None, _adjacent_env(-0.1)) is True
    assert VOIAbstain().should_abstain(None, None, None, _adjacent_env(-1.0)) is False


def test_make_abstain_policy_names():
    assert isinstance(make_abstain_policy("never"), NeverAbstain)
    assert isinstance(make_abstain_policy("entropy"), EntropyAbstain)
    assert isinstance(make_abstain_policy("voi"), VOIAbstain)
    with pytest.raises(ValueError):
        make_abstain_policy("nope")


def test_collect_episodes_with_abstain():
    episodes = collect_episodes(
        10,
        seed=0,
        policy="scripted",
        max_steps=12,
        enable_abstain=True,
        abstain_prob=1.0,
        abstain_penalty=-0.1,
    )

    assert len(episodes) == 10
    assert all(ep["act"].max() < N_ACTIONS_WITH_ABSTAIN for ep in episodes)
    # 至少有一局真的用了第 6 个动作，确认动作确实进入了数据
    assert any((ep["act"] == ABSTAIN_ACTION).any() for ep in episodes)
