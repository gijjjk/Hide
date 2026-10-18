"""
询问与弃权策略。

询问：
- NeverAsk：从不询问；
- FixedAsk：第一步必问；
- EntropyGate：信念熵超过阈值才问，且每局最多问一次；
- VOIGate：期望试错成本高于询问成本才问；
- LearnedAsk：用世界模型的 ask_head（输入潜状态和询问成本）决定是否询问。

弃权（仅在 env.enable_abstain=True 的 6 动作模式下生效）：
- NeverAbstain / EntropyAbstain / VOIAbstain。

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


def _require_belief_supervision(model) -> None:
    """熵/VOI 门控要求信念头用 goal 标签训练过。"""
    if not getattr(model, "use_belief_supervision", True):
        raise RuntimeError(
            "该 checkpoint 的信念头没有用 goal 标签训练；熵/VOI 门控不可用"
        )


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
        _require_belief_supervision(model)
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
        _require_belief_supervision(model)
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


class LearnedAsk:
    """用模型自带的 ask_head 决定是否询问。

    ask_head 的输入是潜状态 (h,z) 和当前询问成本 |ask_cost|；
    训练标签由 data.ask_label() 根据当前状态生成。
    """

    name = "learned"

    def __init__(self, threshold: float = 0.5, min_steps: int = 0):
        self.threshold = threshold
        self.min_steps = min_steps

    def should_ask(self, model, h, z, env) -> bool:
        if env.ask_count > 0:
            return False
        if env.steps < self.min_steps:
            return False
        if getattr(model, "ask_head", None) is None:
            raise RuntimeError(
                "checkpoint 没有 ask_head；请用 steps/02_world_model.py --learned-ask 重新训练"
            )
        cost = torch.tensor([abs(env.ask_cost)], dtype=h.dtype, device=h.device)
        prob = float(model.ask_prob(h, z, cost).item())
        return prob > self.threshold

    def __repr__(self):
        return f"LearnedAsk(threshold={self.threshold}, min_steps={self.min_steps})"


def make_ask_policy(name: str, threshold: float = 0.5, min_steps: int = 0):
    if name == "never":
        return NeverAsk()
    if name == "fixed":
        return FixedAsk()
    if name == "entropy":
        return EntropyGate(threshold, min_steps)
    if name == "voi":
        return VOIGate()
    if name == "learned":
        return LearnedAsk(threshold, min_steps)
    raise ValueError(f"未知的询问策略：{name}")


# ---------- ABSTAIN 门控（任务 2） ----------
#
# ABSTAIN 是“显式放弃本局”的动作：立即结束、不拿成功奖励、不触发错门罚分，
# 只付 abstain_penalty。它只在 env.enable_abstain=True（6 动作）时可用。
# 三个门控都遵守同一个前提：已经问过或已经读过告示牌时不再弃权——
# 信息已经买到，弃权等于把询问成本白白扔掉。


def _can_abstain(env, min_steps: int = 0) -> bool:
    """ABSTAIN 的基本前提：已启用、还没问过、步数够、正站在未知门旁边。"""
    if not getattr(env, "enable_abstain", False):
        return False
    if env.done or env.steps < min_steps:
        return False
    if env.ask_count > 0:
        return False
    r, c = env.agent_pos
    for k, (dr, dc) in enumerate(env.door_positions):
        if env.door_status[k] != -1 and abs(r - dr) + abs(c - dc) == 1:
            return True
    return False


class NeverAbstain:
    name = "never"

    def should_abstain(self, model, h, z, env) -> bool:
        return False


class EntropyAbstain:
    """信念熵超过阈值、且已经走到一扇未知门旁边时弃权。"""

    name = "entropy"

    def __init__(self, threshold: float = 0.5, min_steps: int = 0):
        self.threshold = threshold
        self.min_steps = min_steps

    def should_abstain(self, model, h, z, env) -> bool:
        if not _can_abstain(env, self.min_steps):
            return False
        _require_belief_supervision(model)
        probs = belief_probs(model, h, z)
        return belief_entropy(probs) > self.threshold

    def __repr__(self):
        return f"EntropyAbstain(threshold={self.threshold}, min_steps={self.min_steps})"


class VOIAbstain:
    """期望试错成本高于弃权代价时弃权。

    沿用 VOIGate 的粗略估计：首选门的概率为 p_max 时，
    走错门的期望次数 ≈ (1 - p_max) / p_max。
    弃权代价是 |abstain_penalty|；继续尝试还要付步数代价和错门罚分。
    """

    name = "voi"

    def __init__(self, margin: float = 0.0):
        self.margin = margin

    def should_abstain(self, model, h, z, env) -> bool:
        if not _can_abstain(env, 0):
            return False
        _require_belief_supervision(model)
        probs = belief_probs(model, h, z)
        p_max = max(float(probs.max().item()), 0.05)
        expected_wrong_tries = (1.0 - p_max) / p_max
        expected_try_cost = expected_wrong_tries * (
            abs(env.wrong_penalty) + abs(env.step_penalty)
        )
        abstain_cost = abs(getattr(env, "abstain_penalty", -0.1))
        return expected_try_cost > abstain_cost + self.margin

    def __repr__(self):
        return f"VOIAbstain(margin={self.margin})"


def make_abstain_policy(
    name: str, threshold: float = 0.5, min_steps: int = 0, margin: float = 0.0
):
    if name == "never":
        return NeverAbstain()
    if name == "entropy":
        return EntropyAbstain(threshold, min_steps)
    if name == "voi":
        return VOIAbstain(margin)
    raise ValueError(f"未知的弃权策略：{name}")


def max_entropy() -> float:
    return math.log(3.0)
