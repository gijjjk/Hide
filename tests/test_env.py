"""
环境自检。运行方式（在 D:\\hide 下）：
    python tests/test_env.py

这些测试不需要 PyTorch，只需要 numpy。
"""

import sys
from pathlib import Path

# 让脚本从 tests/ 目录运行时也能 import 到项目根目录的 askdreamer 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 默认可能是 GBK，这里统一成 UTF-8，避免中文/符号乱码
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

from askdreamer.env import MSG_DIM, N_ACTIONS, OBS_DIM, HiddenGoalGrid


def test_shapes():
    env = HiddenGoalGrid(seed=0)
    obs, msg = env.reset()
    assert obs.shape == (OBS_DIM,), obs.shape
    assert msg.shape == (MSG_DIM,), msg.shape
    assert obs.dtype == np.float32
    print("✓ 观测/消息维度正确")


def test_seed_reproducible():
    env_a = HiddenGoalGrid(seed=0)
    env_b = HiddenGoalGrid(seed=0)
    obs_a, msg_a = env_a.reset(seed=123)
    obs_b, msg_b = env_b.reset(seed=123)
    assert np.array_equal(obs_a, obs_b)
    assert np.array_equal(msg_a, msg_b)

    actions = [3, 3, 4, 0, 2, 3, 1, 3]
    for a in actions:
        out_a = env_a.step(a)
        out_b = env_b.step(a)
        assert np.array_equal(out_a[0], out_b[0])
        assert np.array_equal(out_a[1], out_b[1])
        assert out_a[2] == out_b[2]
        assert out_a[3] == out_b[3]
        if out_a[3]:
            break
    print("✓ 相同 seed + 相同动作，结果完全一致")


def test_ask_message():
    env = HiddenGoalGrid(seed=0)
    env.reset(seed=1)
    env.goal = 2
    obs, msg, reward, done, info = env.step(4)  # ASK
    assert msg[3 + 2] == 1.0, msg
    assert info["asked"] is True
    assert reward < 0
    assert done is False
    assert info["ask_count"] == 1
    print("✓ ASK 会把正确答案写进消息通道，并产生代价")


def test_correct_door():
    env = HiddenGoalGrid(seed=0)
    env.reset(seed=2)
    env.goal = 0
    env.agent_pos = (1, 5)          # 0 号门在 (1,6)，从左边一格向右走
    obs, msg, reward, done, info = env.step(3)  # right
    assert info["success"] is True
    assert done is True
    assert reward > 0.9
    print("✓ 走到正确的门会获得 +1 并结束")


def test_wrong_door():
    env = HiddenGoalGrid(seed=0)
    env.reset(seed=3)
    env.goal = 0
    env.agent_pos = (3, 5)          # 1 号门在 (3,6)，但正确门是 0
    obs, msg, reward, done, info = env.step(3)  # right
    assert info["success"] is False
    assert done is False
    assert env.door_status[1] == -1
    assert env.agent_pos == (3, 5)  # 被弹回原地
    assert reward < -0.2
    print("✓ 走错门会扣分、门变成墙、智能体留在原地")


def test_no_goal_leak():
    """观测里不能有 goal；否则信念头会作弊。"""
    env_a = HiddenGoalGrid(seed=0)
    env_b = HiddenGoalGrid(seed=0)
    obs_a, msg_a = env_a.reset(seed=7)
    obs_b, msg_b = env_b.reset(seed=7)
    env_a.goal = 0
    env_b.goal = 2
    assert np.array_equal(env_a.observe(), env_b.observe())
    assert np.array_equal(msg_a, msg_b)
    print("✓ 观测和消息通道都不包含 goal（无泄漏）")


def test_sign():
    env = HiddenGoalGrid(seed=0)
    env.reset(seed=4)
    env.goal = 0
    env.sign_pos = (1, 2)   # 起点 (1,1) 的右边
    env.sign_read = False
    env.msg[:] = 0.0
    obs, msg, reward, done, info = env.step(3)  # 走到告示牌上
    assert info["sign_read"] is True
    assert env.sign_read is True
    assert msg[:3].sum() == 1.0, msg
    eliminated = int(np.argmax(msg[:3]))
    assert eliminated != env.goal
    print("✓ 告示牌会消去一扇错误的门")


def test_timeout():
    env = HiddenGoalGrid(seed=0, max_steps=1)
    env.reset(seed=5)
    obs, msg, reward, done, info = env.step(0)  # 向上撞墙，但会消耗一步
    assert done is True
    assert info["timeout"] is True
    assert reward < 0
    print("✓ 超时判定正常")


def test_boundary_and_invalid_action():
    env = HiddenGoalGrid(seed=0)
    env.reset(seed=6)
    try:
        env.step(N_ACTIONS)
        raise AssertionError("越界动作应该报错")
    except ValueError:
        pass

    env.reset(seed=6)
    env.agent_pos = (1, 1)
    before = env.agent_pos
    env.step(0)  # 向上走到 (0,1)，这是墙
    assert env.agent_pos == before
    print("✓ 越界动作报错，撞墙原地不动")


if __name__ == "__main__":
    test_shapes()
    test_seed_reproducible()
    test_ask_message()
    test_correct_door()
    test_wrong_door()
    test_no_goal_leak()
    test_sign()
    test_timeout()
    test_boundary_and_invalid_action()
    print("\n全部环境测试通过。")
