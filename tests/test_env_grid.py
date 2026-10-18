"""
网格参数化回归测试。运行方式（在 D:\\hide 下）：
    python tests/test_env_grid.py
或
    python -m pytest tests/test_env_grid.py -q

重点：默认 7x7 的行为必须和旧版完全一致；10x10 只改变布局相关逻辑，
观测维度仍然是 150。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

from askdreamer.data import collect_episodes
from askdreamer.env import (
    DOOR_POSITIONS,
    GRID_SIZE,
    OBS_DIM,
    START_POS,
    HiddenGoalGrid,
)


def test_default_layout_unchanged():
    env = HiddenGoalGrid(seed=0)
    assert GRID_SIZE == 7
    assert DOOR_POSITIONS == ((1, 6), (3, 6), (5, 6))
    assert START_POS == (1, 1)
    assert env.grid_size == 7
    assert env.door_positions == DOOR_POSITIONS
    assert env.start_pos == START_POS
    obs, msg = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert msg.shape == (env.msg_dim,)
    print("✓ 默认 7x7 布局与旧常量一致")


def test_larger_grid_observation_and_sign():
    env = HiddenGoalGrid(seed=0, grid_size=10)
    assert env.door_positions == ((1, 9), (5, 9), (8, 9))

    for seed in range(100):
        obs, msg = env.reset(seed=seed)
        assert obs.shape == (OBS_DIM,)
        assert msg.shape == (env.msg_dim,)
        assert env.sign_pos != env.start_pos
        assert env.sign_pos not in env.door_positions
        assert 0 < env.sign_pos[0] < env.grid_size - 1
        assert 0 < env.sign_pos[1] < env.grid_size - 1

    assert len(env.render().splitlines()) == 10
    assert len(env.render().splitlines()[0]) == 10
    print("✓ 10x10 观测仍为 150 维，告示牌不落在起点/门上")


def test_timeout_uses_instance_max_steps():
    env = HiddenGoalGrid(seed=0, grid_size=10, max_steps=7)
    env.reset(seed=0)

    done = False
    info = {}
    for _ in range(7):
        _, _, _, done, info = env.step(3)  # 一直向右走
        if done:
            break

    assert done
    assert env.steps == 7
    assert info["timeout"] is True
    print("✓ 10x10 下 max_steps 正确截断")


def test_invalid_layout_rejected():
    for kwargs in (
        {"grid_size": 4},
        {"grid_size": 10, "door_positions": ((1, 9), (5, 9))},
        {"grid_size": 10, "door_positions": ((1, 9), (1, 9), (8, 9))},
        {"grid_size": 10, "start_pos": (1, 9)},
    ):
        try:
            HiddenGoalGrid(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应当拒绝非法布局：{kwargs}")
    print("✓ 非法网格/门/起点会被拒绝")


def test_collect_episodes_grid_size():
    episodes = collect_episodes(
        2, seed=0, policy="scripted", max_steps=6, grid_size=10
    )
    assert len(episodes) == 2
    assert episodes[0]["obs"].shape == (6, OBS_DIM)
    assert episodes[0]["msg"].shape == (6, 6)
    assert episodes[0]["act"].shape == (6,)
    assert np.all(episodes[0]["valid"] >= 0)
    print("✓ collect_episodes 支持 10x10 且缓冲维度不变")


if __name__ == "__main__":
    test_default_layout_unchanged()
    test_larger_grid_observation_and_sign()
    test_timeout_uses_instance_max_steps()
    test_invalid_layout_rejected()
    test_collect_episodes_grid_size()
    print("\n全部通过。")
