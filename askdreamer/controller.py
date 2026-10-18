"""
询问策略（v1 规则版）。

- NeverAsk：从不询问；
- FixedAsk：第一步必问；
- EntropyGate：信念熵超过阈值才问，且每局最多问一次。

信念来自世界模型的 belief_head。熵最大是 ln(3)≈1.099（完全不知道哪扇门对），
告示牌排除一扇门后最多是 ln(2)≈0.693。
"""

from __future__ import annotations

import math

import torch

from .model import WorldModel


@torch.no_grad()
def belief_probs(model: WorldModel, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """返回 (1,3) 的信念分布。"""
    logits = model.belief_head(torch.cat([h, z], dim=-1))
    return torch.softmax(logits, dim=-1)


def belief_entropy(probs: torch.Tensor) -> float:
    """香农熵（nat）。"""
    return float(-(probs * torch.log(probs + 1e-8)).sum(dim=-1).item())


class NeverAsk:
    name = "never"

    def should_ask(self, model, h, z, env) -> bool:
        return False


class FixedAsk:
    name = "fixed"

    def should_ask(self, model, h, z, env) -> bool:
        # 第一次被调用时就问；配合 sign_first 时，会先读告示牌再问
        return env.ask_count == 0


class EntropyGate:
    name = "entropy"

    def __init__(self, threshold: float = 0.5, min_steps: int = 0):
        self.threshold = threshold
        self.min_steps = min_steps

    def should_ask(self, model, h, z, env) -> bool:
        if env.ask_count > 0:
            return False
        if env.steps < self.min_steps:
            return False
        probs = belief_probs(model, h, z)
        return belief_entropy(probs) > self.threshold

    def __repr__(self):
        return f"EntropyGate(threshold={self.threshold}, min_steps={self.min_steps})"


class VOIGate:
    """按“信息价值 vs 询问成本”决策的规则版门控。

    粗略估计：如果信念把概率最大的门当作首选，走错门的期望次数是
        (1 - p_max) / p_max
    每次走错要付 wrong_penalty + 步数代价；询问要付 ask_cost。
    只有“不问的期望试错成本” > “询问成本”时才问。
    """

    name = "voi"

    def __init__(self, margin: float = 0.0):
        self.margin = margin

    def should_ask(self, model, h, z, env) -> bool:
        if env.ask_count > 0:
            return False
        probs = belief_probs(model, h, z)
        p_max = max(float(probs.max().item()), 0.05)
        expected_wrong_tries = (1.0 - p_max) / p_max
        expected_try_cost = expected_wrong_tries * (
            abs(env.wrong_penalty) + abs(env.step_penalty)
        )
        ask_cost = abs(env.ask_cost)
        return expected_try_cost > ask_cost + self.margin

    def __repr__(self):
        return f"VOIGate(margin={self.margin})"


def _move_towards(pos, target) -> int:
    """贪心走法：先上下对齐，再向右。"""
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
    return 3


class SignFirstController:
    """先读告示牌，再决定是否询问。

    这是一个很小的“探索先验”：告示牌不直接给奖励，但能排除一扇门。
    读完之后，信念熵最多是 ln(2)≈0.693；此时“问不问”才真正取决于询问成本。
    """

    def __init__(self, ask_policy):
        self.ask_policy = ask_policy
        self.name = f"sign_first+{ask_policy.name}"

    def should_ask(self, model, h, z, env) -> bool:
        if not env.sign_read:
            return False
        return self.ask_policy.should_ask(model, h, z, env)

    def override_action(self, env):
        """还没读告示牌时，覆盖规划器的动作，先走过去。"""
        if env.sign_read:
            return None
        if env.agent_pos == env.sign_pos:
            return None
        return _move_towards(env.agent_pos, env.sign_pos)

    def __repr__(self):
        return f"SignFirstController({self.ask_policy!r})"


def make_ask_policy(name: str, threshold: float = 0.5, min_steps: int = 0):
    if name == "never":
        return NeverAsk()
    if name == "fixed":
        return FixedAsk()
    if name == "entropy":
        return EntropyGate(threshold, min_steps)
    if name == "voi":
        return VOIGate()
    raise ValueError(f"未知的询问策略：{name}")


def max_entropy() -> float:
    return math.log(3.0)
