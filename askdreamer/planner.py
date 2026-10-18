"""
潜空间规划器（System 2 的第一版）。

思路：
- 不和环境交互，只在世界模型的 prior 里“想象”；
- 随机采样 N 条长度为 H 的动作序列；
- 用奖励头给每条序列打分，选最高分序列的第一个动作；
- 这就是“在潜空间里思考”，N 就是思考预算。

注意：规划只使用 4 个移动动作，不包含 ASK，也不包含 ABSTAIN；
询问和弃权都由 controller 单独决定，这样“想”和“买信息/退出”是两套机制。
"""

from __future__ import annotations

import torch

from .model import WorldModel


class RandomShootingPlanner:
    def __init__(
        self,
        model: WorldModel,
        horizon: int = 8,
        samples: int = 32,
        gamma: float = 0.99,
        use_value: bool = True,
    ):
        self.model = model
        self.horizon = horizon
        self.samples = samples
        self.gamma = gamma
        self.use_value = use_value

    @torch.no_grad()
    def act(self, h: torch.Tensor, z: torch.Tensor, msg: torch.Tensor, generator: torch.Generator | None = None):
        """h,z,msg: (1, dim)。返回 (动作, 预测回报)。"""
        # 每条候选序列都是 horizon 个 0..3 的移动动作
        actions = torch.randint(
            0, 4, (self.samples, self.horizon), generator=generator
        )

        h_cur = h.repeat(self.samples, 1)
        z_cur = z.repeat(self.samples, 1)
        msg_cur = msg.repeat(self.samples, 1)
        total = torch.zeros(self.samples)

        for k in range(self.horizon):
            a = actions[:, k]
            r = self.model.predict_reward(h_cur, z_cur, a)  # (samples,)
            total = total + (self.gamma ** k) * r
            h_cur, z_cur = self.model.prior_step(h_cur, z_cur, a, msg_cur)

        # 最后一步之后的价值：即使用 H 步还没走到门，也能判断“离成功有多近”
        if self.use_value:
            v = self.model.value_head(torch.cat([h_cur, z_cur], dim=-1)).squeeze(-1)
            total = total + (self.gamma ** self.horizon) * v

        best = int(total.argmax().item())
        return int(actions[best, 0].item()), float(total[best].item())
