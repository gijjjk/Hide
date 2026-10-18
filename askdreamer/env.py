"""
HiddenGoalGrid：部分可观测的隐藏目标网格环境。

坐标约定：使用 (row, col)，左上角是 (0, 0)。row 向下增大，col 向右增大。
"""

from __future__ import annotations

import numpy as np

# ---------- 可以被其他文件 import 的常量 ----------
GRID_SIZE = 7          # 7x7 网格，最外圈是墙
VIEW_SIZE = 5          # 智能体只能看到以自己为中心的 5x5 局部视野
N_CHANNELS = 6         # 每个格子的 6 个通道：墙、门0、门1、门2、告示牌、自己
OBS_DIM = VIEW_SIZE * VIEW_SIZE * N_CHANNELS   # 150
MSG_DIM = 6            # 消息通道：前 3 维=告示牌消去的门，后 3 维=oracle 答案
N_ACTIONS = 5          # 0..3 移动，4=ASK
ACTION_NAMES = ("up", "down", "left", "right", "ASK")
ACTION_NAMES_WITH_ABSTAIN = ACTION_NAMES + ("ABSTAIN",)

# ABSTAIN 只在 enable_abstain=True 时启用的第 6 个动作。
# 单独保留 N_ACTIONS=5，旧 checkpoint / 旧命令的行为完全不变。
ASK_ACTION = 4
ABSTAIN_ACTION = 5
N_ACTIONS_WITH_ABSTAIN = 6


def _default_door_positions(grid_size: int):
    """三扇门固定在地图右侧；g=7 时正好是 ((1,6),(3,6),(5,6))。"""
    return (
        (1, grid_size - 1),
        (grid_size // 2, grid_size - 1),
        (grid_size - 2, grid_size - 1),
    )


# 三扇门固定在地图右侧；门本身在墙的位置，只有从左边一格才能走上去。
# 保留模块级默认值，旧代码 `from askdreamer.env import DOOR_POSITIONS` 不受影响。
DOOR_POSITIONS = _default_door_positions(GRID_SIZE)
START_POS = (1, 1)

# 移动方向：上、下、左、右
_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


class HiddenGoalGrid:
    """隐藏目标网格环境。

    每一局随机选一扇“正确的门” goal ∈ {0,1,2}。
    智能体看不到 goal；它可以：
      1. 走到告示牌 S 上，消去一扇错误的门（消息通道前 3 维）；
      2. 花一步 + 询问代价执行 ASK，oracle 直接告诉它正确的门（消息通道后 3 维）；
      3. 走到某扇门上：正确则 +1 并结束，错误则 -0.3 且该门变成墙；
      4. 如果 enable_abstain=True，还可以执行 ABSTAIN（动作 5）：立即结束本局，
         不拿成功奖励、不触发错门罚分，只付 abstain_penalty。

    注意：goal 不会进入 150 维观测。它只出现在 info["goal"] 里，
    训练世界模型时可以把它当作“特权标签”来监督信念头，但测试时智能体只能靠历史推断。
    """

    def __init__(
        self,
        seed: int = 0,
        max_steps: int = 40,
        ask_cost: float = -0.05,
        step_penalty: float = -0.01,
        wrong_penalty: float = -0.3,
        timeout_penalty: float = -0.1,
        grid_size: int = GRID_SIZE,
        door_positions=None,
        start_pos: tuple[int, int] = START_POS,
        enable_abstain: bool = False,
        abstain_penalty: float = -0.1,
    ):
        if grid_size < 5:
            raise ValueError(f"grid_size 至少为 5，收到 {grid_size}")

        doors = tuple(
            _default_door_positions(grid_size)
            if door_positions is None
            else door_positions
        )
        doors = tuple(tuple(pos) for pos in doors)
        if len(doors) != 3:
            raise ValueError(f"门的位置必须恰好 3 个，收到 {len(doors)} 个")
        if len(set(doors)) != 3:
            raise ValueError(f"门的位置不能重复：{doors}")
        if tuple(start_pos) in doors:
            raise ValueError(f"起点 {start_pos} 不能和门重合")
        for r, c in doors:
            if not (0 <= r < grid_size and 0 <= c < grid_size):
                raise ValueError(f"门 {r, c} 超出 {grid_size}x{grid_size} 地图")
        if not (0 <= start_pos[0] < grid_size and 0 <= start_pos[1] < grid_size):
            raise ValueError(f"起点 {start_pos} 超出 {grid_size}x{grid_size} 地图")

        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.ask_cost = ask_cost
        self.step_penalty = step_penalty
        self.wrong_penalty = wrong_penalty
        self.timeout_penalty = timeout_penalty
        self.grid_size = grid_size
        self.door_positions = doors
        self.start_pos = tuple(start_pos)
        self.enable_abstain = enable_abstain
        self.abstain_penalty = abstain_penalty
        self.obs_dim = OBS_DIM
        self.msg_dim = MSG_DIM
        self.n_actions = N_ACTIONS_WITH_ABSTAIN if enable_abstain else N_ACTIONS

        # 下面这些变量会在 reset() 里被重新赋值
        self.goal = 0
        self.agent_pos = self.start_pos
        self.door_status = [0, 0, 0]   # 0=未知/正常, -1=已证明错误
        self.sign_pos = self.start_pos
        self.sign_read = False
        self.msg = np.zeros(self.msg_dim, dtype=np.float32)
        self.steps = 0
        self.done = False
        self.ask_count = 0
        self.abstained = False

    # ---------- 基本查询 ----------
    def door_at(self, row: int, col: int):
        """如果 (row,col) 是一扇门，返回门编号 0/1/2；否则返回 None。"""
        for k, pos in enumerate(self.door_positions):
            if (row, col) == pos:
                return k
        return None

    def is_wall(self, row: int, col: int) -> bool:
        """判断一个格子是不是墙（或被证明错误的门）。"""
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return True

        door = self.door_at(row, col)
        if door is not None:
            # 错误的门变成墙；未知/正确的门可以走上去
            return self.door_status[door] == -1

        # 最外圈是墙
        return (
            row == 0
            or row == self.grid_size - 1
            or col == 0
            or col == self.grid_size - 1
        )

    # ---------- 观测 ----------
    def observe(self) -> np.ndarray:
        """返回 150 维局部观测。goal 绝不会出现在这里。"""
        view = np.zeros((VIEW_SIZE, VIEW_SIZE, N_CHANNELS), dtype=np.float32)
        center_r, center_c = self.agent_pos

        for vr in range(VIEW_SIZE):
            for vc in range(VIEW_SIZE):
                r = center_r + vr - VIEW_SIZE // 2
                c = center_c + vc - VIEW_SIZE // 2

                if self.is_wall(r, c):
                    view[vr, vc, 0] = 1.0

                door = self.door_at(r, c)
                if door is not None and self.door_status[door] != -1:
                    view[vr, vc, 1 + door] = 1.0

                if (r, c) == self.sign_pos and not self.sign_read:
                    view[vr, vc, 4] = 1.0

                if (r, c) == self.agent_pos:
                    view[vr, vc, 5] = 1.0

        return view.reshape(-1)

    # ---------- 回合开始 ----------
    def reset(self, seed: int | None = None):
        """开始新的一局。返回 (obs, msg)。"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.goal = int(self.rng.integers(0, 3))
        self.agent_pos = self.start_pos
        self.door_status = [0, 0, 0]
        self.sign_read = False
        self.msg = np.zeros(self.msg_dim, dtype=np.float32)
        self.steps = 0
        self.done = False
        self.ask_count = 0
        self.abstained = False

        # 告示牌随机放在内部格子，但不能和起点或门重合
        excluded = set(self.door_positions)
        candidates = [
            (r, c)
            for r in range(1, self.grid_size - 1)
            for c in range(1, self.grid_size - 1)
            if (r, c) != self.start_pos and (r, c) not in excluded
        ]
        if not candidates:
            raise ValueError("没有可放告示牌的内部格子")
        self.sign_pos = candidates[int(self.rng.integers(0, len(candidates)))]

        return self.observe(), self.msg.copy()

    # ---------- 与环境交互一步 ----------
    def step(self, action: int):
        """执行动作。返回 (obs, msg, reward, done, info)。"""
        if self.done:
            raise RuntimeError("这一局已经结束，请先调用 reset()。")
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"动作必须是 0..{self.n_actions - 1}，收到 {action}")

        self.steps += 1
        reward = self.step_penalty
        info = {
            "asked": False,
            "sign_read": False,
            "success": False,
            "timeout": False,
            "abstained": False,
        }

        if action == ASK_ACTION:
            # ASK：不移动，花一步和一点奖励，oracle 把正确门写进消息通道
            self.ask_count += 1
            self.msg[3 + self.goal] = 1.0
            reward += self.ask_cost
            info["asked"] = True
        elif action == ABSTAIN_ACTION:
            # ABSTAIN：显式放弃本局，避免冒险撞错门，但拿不到成功奖励。
            self.abstained = True
            self.done = True
            reward += self.abstain_penalty
            info["abstained"] = True
        else:
            dr, dc = _MOVES[action]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            door = self.door_at(nr, nc)

            if door is not None and self.door_status[door] != -1:
                if door == self.goal:
                    # 找到正确的门
                    reward += 1.0
                    self.done = True
                    info["success"] = True
                else:
                    # 错误的门：扣分，并把门变成墙，智能体留在原地
                    reward += self.wrong_penalty
                    self.door_status[door] = -1
            elif self.is_wall(nr, nc):
                # 撞墙或被证明错误的门：留在原地
                pass
            else:
                # 正常移动
                self.agent_pos = (nr, nc)

                # 如果走到告示牌上，消去一扇错误的门
                if (nr, nc) == self.sign_pos and not self.sign_read:
                    self.sign_read = True
                    info["sign_read"] = True
                    wrong_doors = [
                        k for k in range(len(self.door_positions)) if k != self.goal
                    ]
                    eliminated = wrong_doors[int(self.rng.integers(0, len(wrong_doors)))]
                    self.msg[eliminated] = 1.0

        # 超时
        if self.steps >= self.max_steps and not self.done:
            reward += self.timeout_penalty
            self.done = True
            info["timeout"] = True

        # 这些信息只用于调试/训练标签，不是智能体的观测
        info["goal"] = self.goal
        info["steps"] = self.steps
        info["agent_pos"] = self.agent_pos
        info["door_status"] = list(self.door_status)
        info["ask_count"] = self.ask_count
        info["sign_pos"] = self.sign_pos
        info["sign_read"] = self.sign_read
        info["abstained"] = self.abstained

        return self.observe(), self.msg.copy(), float(reward), self.done, info

    # ---------- 可视化 ----------
    def render(self, show_goal: bool = False) -> str:
        """返回 ASCII 地图。show_goal=True 仅用于你自己调试，智能体看不到。"""
        lines = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                if (r, c) == self.agent_pos:
                    ch = "A"
                elif self.door_at(r, c) is not None:
                    k = self.door_at(r, c)
                    if self.door_status[k] == -1:
                        ch = "#"
                    elif show_goal and k == self.goal:
                        ch = "*"
                    else:
                        ch = str(k)
                elif (r, c) == self.sign_pos and not self.sign_read:
                    ch = "S"
                elif self.is_wall(r, c):
                    ch = "#"
                else:
                    ch = "."
                row.append(ch)
            lines.append("".join(row))
        return "\n".join(lines)
