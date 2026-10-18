"""
张量形状 / 前向 / 反向 / 规划器 / 控制器自检。

运行方式（在 D:\\hide 下）：
    python tests/test_shapes.py

纯 assert，不需要 pytest。这个测试不追求模型效果，只确认所有 shape 对得上、
梯度能回传、规划器返回合法动作。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import torch

from askdreamer.controller import (
    EntropyGate,
    FixedAsk,
    NeverAsk,
    SignFirstController,
    VOIGate,
    belief_entropy,
    belief_probs,
)
from askdreamer.env import MSG_DIM, N_ACTIONS, OBS_DIM
from askdreamer.model import WorldModel
from askdreamer.planner import RandomShootingPlanner

torch.manual_seed(0)
torch.set_num_threads(3)


def make_batch(B: int = 3, L: int = 5) -> dict:
    """构造一个形状正确的小 batch，不依赖真实环境。"""
    return {
        "obs": torch.randn(B, L, OBS_DIM),
        "msg": torch.randn(B, L, MSG_DIM),
        "act": torch.randint(0, N_ACTIONS, (B, L)),
        "rew": torch.randn(B, L) * 0.1,
        "done": torch.zeros(B, L),
        "goal": torch.randint(0, 3, (B, L)),
        "valid": torch.ones(B, L),
        "next_obs": torch.randn(B, L, OBS_DIM),
        "valid_next": torch.ones(B, L),
    }


def test_forward_backward():
    batch = make_batch()
    for use_decoder in (True, False):
        model = WorldModel(use_decoder=use_decoder)
        out = model.encode_sequence(batch["obs"], batch["msg"], batch["act"])
        B, L = batch["obs"].shape[:2]
        assert out["e"].shape == (B, L, model.e_dim), out["e"].shape
        assert out["h"].shape == (B, L, model.h_dim), out["h"].shape
        assert out["z"].shape == (B, L, model.z_dim), out["z"].shape
        assert out["prior"][0].shape == (B, L, model.z_dim)
        assert out["posterior"][0].shape == (B, L, model.z_dim)

        loss, metrics = model.compute_losses(batch)
        assert torch.isfinite(loss), loss
        loss.backward()
        grad_norm = sum(
            float(p.grad.abs().sum())
            for p in model.parameters()
            if p.grad is not None
        )
        assert grad_norm > 0.0, "没有任何梯度回传"
        print(
            f"✓ forward/backward (decoder={use_decoder}) "
            f"loss={loss.item():.4f} grad_sum={grad_norm:.3f}"
        )


def test_single_step_shapes():
    model = WorldModel(use_decoder=False)
    h = torch.zeros(1, model.h_dim)
    z = torch.zeros(1, model.z_dim)
    msg = torch.zeros(1, MSG_DIM)
    obs = torch.randn(1, OBS_DIM)

    h1, z1, prior, post = model.observe_step(h, z, None, msg, obs)
    assert h1.shape == (1, model.h_dim), h1.shape
    assert z1.shape == (1, model.z_dim), z1.shape
    assert prior[0].shape == (1, model.z_dim)
    assert post[0].shape == (1, model.z_dim)

    action = torch.tensor([0])
    h2, z2 = model.prior_step(h1, z1, action, msg)
    assert h2.shape == (1, model.h_dim), h2.shape
    assert z2.shape == (1, model.z_dim), z2.shape

    r = model.predict_reward(h2, z2, action)
    assert r.shape == (1,), r.shape
    print("✓ observe_step / prior_step / predict_reward 形状正确")


def test_planner_and_controller():
    model = WorldModel(use_decoder=False)
    model.eval()
    planner = RandomShootingPlanner(model, horizon=3, samples=4)
    h = torch.zeros(1, model.h_dim)
    z = torch.zeros(1, model.z_dim)
    msg = torch.zeros(1, MSG_DIM)

    action, score = planner.act(h, z, msg)
    assert isinstance(action, int)
    assert 0 <= action < 4, action          # 规划器只选移动动作，不含 ASK
    assert isinstance(score, float)

    probs = belief_probs(model, h, z)
    assert probs.shape == (1, 3), probs.shape
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)
    entropy = belief_entropy(probs)
    assert 0.0 <= entropy <= 1.0987, entropy   # ln(3) ≈ 1.0986

    class FakeEnv:
        ask_count = 0
        steps = 0
        ask_cost = -0.05
        wrong_penalty = -0.3
        step_penalty = -0.01
        sign_read = False
        agent_pos = (1, 1)
        sign_pos = (1, 2)

    env = FakeEnv()
    assert NeverAsk().should_ask(model, h, z, env) is False
    assert FixedAsk().should_ask(model, h, z, env) is True
    env.ask_count = 1
    assert FixedAsk().should_ask(model, h, z, env) is False
    env.ask_count = 0

    assert EntropyGate(threshold=99.0).should_ask(model, h, z, env) is False
    assert EntropyGate(threshold=-1.0).should_ask(model, h, z, env) is True
    assert isinstance(VOIGate().should_ask(model, h, z, env), bool)

    sign_first = SignFirstController(FixedAsk())
    assert sign_first.should_ask(model, h, z, env) is False   # 还没读牌
    override = sign_first.override_action(env)
    assert override in (0, 1, 2, 3), override

    print("✓ planner / belief / ask gate 自检通过")


def test_parameter_budget():
    nodec = WorldModel(use_decoder=False)
    dec = WorldModel(use_decoder=True)
    n_nodec = sum(p.numel() for p in nodec.parameters() if p.requires_grad)
    n_dec = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    assert n_nodec < 500_000, n_nodec
    assert n_dec < 500_000, n_dec
    assert n_nodec < n_dec
    print(f"✓ 参数量：无解码器 {n_nodec:,} < 有解码器 {n_dec:,} < 50 万")


if __name__ == "__main__":
    test_forward_backward()
    test_single_step_shapes()
    test_planner_and_controller()
    test_parameter_budget()
    print("\n全部形状测试通过。")
