"""
数据收集与序列采样。

包含脚本策略、episode 收集和固定长度训练序列采样。
"""

from __future__ import annotations

import numpy as np

from .env import (
    ABSTAIN_ACTION,
    GRID_SIZE,
    MSG_DIM,
    OBS_DIM,
    START_POS,
    HiddenGoalGrid,
)


class ScriptedPolicy:
    """按固定规则收集数据，覆盖走到门、询问和读取告示牌三类行为。

    规则：
    - 第一步以 ask_prob 的概率 ASK；
    - 以 sign_prob 的概率先绕去告示牌；
    - 否则走向一扇还没被证明错误的门。
    """

    def __init__(
        self,
        rng: np.random.Generator,
        ask_prob: float = 0.5,
        sign_prob: float = 0.3,
        abstain_prob: float = 0.0,
    ):
        self.rng = rng
        self.ask_prob = ask_prob
        self.visit_sign = bool(rng.random() < sign_prob)
        self.asked = False
        self.abstain_prob = abstain_prob
        self.abstained = False
        self.target = None

    def act(self, env: HiddenGoalGrid) -> int:
        # 第一步询问
        if not self.asked and env.steps == 0 and self.rng.random() < self.ask_prob:
            self.asked = True
            return 4  # ASK

        # 信息不足且马上要撞门时，按概率选择 ABSTAIN（只在 6 动作模式下）
        if (
            env.enable_abstain
            and self.abstain_prob > 0
            and not self.abstained
            and env.ask_count == 0
            and not env.sign_read
            and self._adjacent_to_unknown_door(env)
            and self.rng.random() < self.abstain_prob
        ):
            self.abstained = True
            return ABSTAIN_ACTION

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
    def _adjacent_to_unknown_door(env: HiddenGoalGrid) -> bool:
        """智能体是否正站在某扇尚未排除的门旁边。"""
        r, c = env.agent_pos
        for k, (dr, dc) in enumerate(env.door_positions):
            if env.door_status[k] != -1 and abs(r - dr) + abs(c - dc) == 1:
                return True
        return False

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


def ask_label(env: HiddenGoalGrid) -> float:
    """询问监督标签：在当前状态下，询问的期望收益是否高于询问成本。

    估计方式：剩余 k 扇门时，顺序试门的期望错误次数为 (k-1)/2，
    每次错误付出 wrong_penalty 和一步的 step_penalty。
    已经询问过或只剩一扇门时不再需要询问。
    """
    if env.ask_count > 0:
        return 0.0
    # 已被证明错误（door_status=-1）或已被告示牌排除（msg 前 3 维=1）的门不再计入
    alive = sum(
        1
        for k, status in enumerate(env.door_status)
        if status != -1 and env.msg[k] == 0
    )
    if alive <= 1:
        return 0.0
    expected_wrong_tries = (alive - 1) / 2.0
    expected_try_cost = expected_wrong_tries * (
        abs(env.wrong_penalty) + abs(env.step_penalty)
    )
    return 1.0 if expected_try_cost > abs(env.ask_cost) else 0.0


def collect_episodes(
    n_episodes: int,
    seed: int = 0,
    policy: str = "mixed",
    max_steps: int = 40,
    grid_size: int = GRID_SIZE,
    door_positions=None,
    start_pos=START_POS,
    enable_abstain: bool = False,
    abstain_prob: float = 0.0,
    abstain_penalty: float = -0.1,
    random_abstain_prob: float = 0.05,
    ask_costs: list[float] | None = None,
) -> list[dict]:
    """收集若干局数据，每局补齐到 max_steps，用 valid 标记真实步数。

    policy:
      - "random"   : 纯随机动作（包含 ASK；启用 ABSTAIN 时按 random_abstain_prob 采样第 6 个动作）
      - "scripted" : 只用脚本策略
      - "mixed"    : 70% 脚本 + 30% 随机，兼顾覆盖和有效行为

    grid_size/door_positions/start_pos 默认值保持旧的 7x7 行为。
    enable_abstain=False 时动作空间仍是 5 个，旧数据和旧 checkpoint 完全兼容。

    ask_costs 非空时，按 episode 轮流使用不同的询问成本，并记录每步的
    ask_label（见 ask_label()），用于训练学习型询问门控。
    """
    env = HiddenGoalGrid(
        seed=seed,
        max_steps=max_steps,
        grid_size=grid_size,
        door_positions=door_positions,
        start_pos=start_pos,
        enable_abstain=enable_abstain,
        abstain_penalty=abstain_penalty,
    )
    rng = np.random.default_rng(seed + 12_345)
    episodes = []

    for episode_idx in range(n_episodes):
        if ask_costs:
            env.ask_cost = float(ask_costs[episode_idx % len(ask_costs)])
        obs, msg = env.reset()

        obs_buf = np.zeros((max_steps, OBS_DIM), dtype=np.float32)
        msg_buf = np.zeros((max_steps, MSG_DIM), dtype=np.float32)
        act_buf = np.zeros((max_steps,), dtype=np.int64)
        rew_buf = np.zeros((max_steps,), dtype=np.float32)
        done_buf = np.zeros((max_steps,), dtype=np.float32)
        goal_buf = np.zeros((max_steps,), dtype=np.int64)
        valid_buf = np.zeros((max_steps,), dtype=np.float32)
        cost_buf = np.zeros((max_steps,), dtype=np.float32)
        label_buf = np.zeros((max_steps,), dtype=np.float32)

        use_scripted = policy == "scripted" or (
            policy == "mixed" and rng.random() < 0.7
        )
        scripted = (
            ScriptedPolicy(rng, abstain_prob=abstain_prob) if use_scripted else None
        )

        for t in range(max_steps):
            if scripted is not None:
                action = scripted.act(env)
            elif (
                enable_abstain
                and random_abstain_prob > 0
                and rng.random() < random_abstain_prob
            ):
                action = ABSTAIN_ACTION
            else:
                # 随机策略默认在移动/ASK 动作中均匀采样；ABSTAIN 单独抽样，
                # 避免随机局过早结束、训练数据过短。
                action = int(rng.integers(0, env.n_actions - (1 if enable_abstain else 0)))

            # 存的是“执行动作前”的观测/消息，以及执行后的奖励
            obs_buf[t] = obs
            msg_buf[t] = msg
            act_buf[t] = action
            goal_buf[t] = env.goal
            valid_buf[t] = 1.0
            cost_buf[t] = env.ask_cost
            label_buf[t] = ask_label(env)

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
                "cost": cost_buf,
                "ask_label": label_buf,
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
        import torch  # 延迟导入，读取环境时不加载 torch

        B, L = batch_size, self.seq_len
        obs = np.zeros((B, L, OBS_DIM), dtype=np.float32)
        msg = np.zeros((B, L, MSG_DIM), dtype=np.float32)
        act = np.zeros((B, L), dtype=np.int64)
        rew = np.zeros((B, L), dtype=np.float32)
        done = np.zeros((B, L), dtype=np.float32)
        goal = np.zeros((B, L), dtype=np.int64)
        valid = np.zeros((B, L), dtype=np.float32)
        cost = np.zeros((B, L), dtype=np.float32)
        ask_label_buf = np.zeros((B, L), dtype=np.float32)
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
            cost[i] = ep["cost"][start : start + L]
            ask_label_buf[i] = ep["ask_label"][start : start + L]

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
            "cost": torch.from_numpy(cost),
            "ask_label": torch.from_numpy(ask_label_buf),
            "next_obs": torch.from_numpy(next_obs),
            "valid_next": torch.from_numpy(valid_next),
        }
