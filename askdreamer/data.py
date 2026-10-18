"""
数据收集与序列采样。

为什么不用 gym / DataLoader？
- 环境很小，自己写更直观；
- 你能清楚看到每一条数据是怎么来的；
- 避免额外依赖和 Windows 上的多进程问题。
"""

from __future__ import annotations

import numpy as np

from .env import (
    GRID_SIZE,
    MSG_DIM,
    N_ACTIONS,
    OBS_DIM,
    START_POS,
    HiddenGoalGrid,
)


class ScriptedPolicy:
    """一个很笨但有用的脚本策略，用来收集“会走到门、会询问、会读告示牌”的数据。

    它不使用 oracle 答案，只是按固定规则行动：
    - 第一步以 ask_prob 的概率 ASK；
    - 以 sign_prob 的概率先绕去告示牌；
    - 否则走向一扇还没被证明错误的门。
    """

    def __init__(self, rng: np.random.Generator, ask_prob: float = 0.5, sign_prob: float = 0.3):
        self.rng = rng
        self.ask_prob = ask_prob
        self.visit_sign = bool(rng.random() < sign_prob)
        self.asked = False
        self.target = None

    def act(self, env: HiddenGoalGrid) -> int:
        # 第一步询问
        if not self.asked and env.steps == 0 and self.rng.random() < self.ask_prob:
            self.asked = True
            return 4  # ASK

        # 先去看告示牌（如果这局决定去的话）
        if self.visit_sign and not env.sign_read:
            return self._move_towards(env.agent_pos, env.sign_pos)

        # 选一扇还没被证明错误的门
        available = [
            k for k in range(len(env.door_positions)) if env.door_status[k] != -1
        ]
        if not available:
            return int(self.rng.integers(0, 4))
        if self.target is None or self.target not in available:
            self.target = int(self.rng.choice(available))
        return self._move_towards(env.agent_pos, env.door_positions[self.target])

    @staticmethod
    def _move_towards(pos, target) -> int:
        """简单贪心：先上下对齐，再向右。地图内部没有障碍，所以够用。"""
        r, c = pos
        tr, tc = target
        if r < tr:
            return 1  # down
        if r > tr:
            return 0  # up
        if c < tc:
            return 3  # right
        if c > tc:
            return 2  # left
        return 3      # 已经到目标附近，尝试向右（门）


def collect_episodes(
    n_episodes: int,
    seed: int = 0,
    policy: str = "mixed",
    max_steps: int = 40,
    grid_size: int = GRID_SIZE,
    door_positions=None,
    start_pos=START_POS,
) -> list[dict]:
    """收集若干局数据，每局补齐到 max_steps，用 valid 标记真实步数。

    policy:
      - "random"   : 纯随机动作（包含 ASK）
      - "scripted" : 只用脚本策略
      - "mixed"    : 70% 脚本 + 30% 随机，兼顾覆盖和有效行为

    grid_size/door_positions/start_pos 默认值保持旧的 7x7 行为。
    """
    env = HiddenGoalGrid(
        seed=seed,
        max_steps=max_steps,
        grid_size=grid_size,
        door_positions=door_positions,
        start_pos=start_pos,
    )
    rng = np.random.default_rng(seed + 12_345)
    episodes = []

    for _ in range(n_episodes):
        obs, msg = env.reset()

        obs_buf = np.zeros((max_steps, OBS_DIM), dtype=np.float32)
        msg_buf = np.zeros((max_steps, MSG_DIM), dtype=np.float32)
        act_buf = np.zeros((max_steps,), dtype=np.int64)
        rew_buf = np.zeros((max_steps,), dtype=np.float32)
        done_buf = np.zeros((max_steps,), dtype=np.float32)
        goal_buf = np.zeros((max_steps,), dtype=np.int64)
        valid_buf = np.zeros((max_steps,), dtype=np.float32)

        use_scripted = policy == "scripted" or (
            policy == "mixed" and rng.random() < 0.7
        )
        scripted = ScriptedPolicy(rng) if use_scripted else None

        for t in range(max_steps):
            action = scripted.act(env) if scripted else int(rng.integers(0, N_ACTIONS))

            # 存的是“执行动作前”的观测/消息，以及执行后的奖励
            obs_buf[t] = obs
            msg_buf[t] = msg
            act_buf[t] = action
            goal_buf[t] = env.goal
            valid_buf[t] = 1.0

            obs, msg, reward, done, info = env.step(action)
            rew_buf[t] = reward
            done_buf[t] = float(done)

            if done:
                break

        episodes.append(
            {
                "obs": obs_buf,
                "msg": msg_buf,
                "act": act_buf,
                "rew": rew_buf,
                "done": done_buf,
                "goal": goal_buf,
                "valid": valid_buf,
            }
        )

    return episodes


class EpisodeDataset:
    """从收集到的 episode 里随机切长度为 seq_len 的窗口。"""

    def __init__(self, episodes: list[dict], seq_len: int = 16, seed: int = 0):
        assert len(episodes) > 0
        self.episodes = episodes
        self.seq_len = seq_len
        self.max_steps = episodes[0]["obs"].shape[0]
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, batch_size: int):
        """返回一个 batch 的 numpy 数组（训练脚本里再转成 torch）。"""
        import torch  # 放在函数里，这样只读 env 时不需要 torch

        B, L = batch_size, self.seq_len
        obs = np.zeros((B, L, OBS_DIM), dtype=np.float32)
        msg = np.zeros((B, L, MSG_DIM), dtype=np.float32)
        act = np.zeros((B, L), dtype=np.int64)
        rew = np.zeros((B, L), dtype=np.float32)
        done = np.zeros((B, L), dtype=np.float32)
        goal = np.zeros((B, L), dtype=np.int64)
        valid = np.zeros((B, L), dtype=np.float32)
        next_obs = np.zeros((B, L, OBS_DIM), dtype=np.float32)
        valid_next = np.zeros((B, L), dtype=np.float32)

        start_max = self.max_steps - L
        for i in range(B):
            while True:
                ep = self.episodes[int(self.rng.integers(0, len(self.episodes)))]
                start = int(self.rng.integers(0, start_max + 1))
                window_valid = ep["valid"][start : start + L]
                # 至少有一个有效步，否则换一个窗口
                if window_valid.sum() >= 1.0:
                    break

            obs[i] = ep["obs"][start : start + L]
            msg[i] = ep["msg"][start : start + L]
            act[i] = ep["act"][start : start + L]
            rew[i] = ep["rew"][start : start + L]
            done[i] = ep["done"][start : start + L]
            goal[i] = ep["goal"][start : start + L]
            valid[i] = window_valid

            # next_obs[t] = obs[t+1]，最后一个位置留 0；valid_next 同步
            if L > 1:
                next_obs[i, :-1] = ep["obs"][start + 1 : start + L]
                valid_next[i, :-1] = ep["valid"][start + 1 : start + L]

        return {
            "obs": torch.from_numpy(obs),
            "msg": torch.from_numpy(msg),
            "act": torch.from_numpy(act),
            "rew": torch.from_numpy(rew),
            "done": torch.from_numpy(done),
            "goal": torch.from_numpy(goal),
            "valid": torch.from_numpy(valid),
            "next_obs": torch.from_numpy(next_obs),
            "valid_next": torch.from_numpy(valid_next),
        }
