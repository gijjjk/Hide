"""
AskDreamer 的世界模型：编码器 + RSSM-lite + 奖励/信念/JEPA 头（可选解码器）。

关键设计：
- 观测 150 维进入编码器；oracle 答案/告示牌走独立的 6 维消息通道进入 GRU。
  这样信念头不能直接从观测里“抄答案”，H3 的实验才有效。
- 主模型不重建观测；`use_decoder=True` 只用于 1a 对照和排查坍塌。
- 潜变量 z 是 16 维高斯；训练用 posterior，想象/规划用 prior。

所有张量 shape 都在注释里标出，方便你调试。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import MSG_DIM, N_ACTIONS, OBS_DIM


class Encoder(nn.Module):
    """把 150 维局部观测压成 64 维嵌入 e。"""

    def __init__(self, obs_dim: int = OBS_DIM, e_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.SiLU(),
            nn.Linear(128, e_dim),
            nn.LayerNorm(e_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, 150) -> (B, 64)
        return self.net(obs)


class RSSM(nn.Module):
    """Recurrent State-Space Model 的简化版。

    h_t = GRU([z_{t-1}, a_{t-1}, msg_t], h_{t-1})
    prior:     p(z_t | h_t)
    posterior: q(z_t | h_t, e_t)
    """

    def __init__(
        self,
        z_dim: int = 16,
        h_dim: int = 96,
        e_dim: int = 64,
        n_actions: int = N_ACTIONS,
        msg_dim: int = MSG_DIM,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.n_actions = n_actions
        self.msg_dim = msg_dim

        self.gru = nn.GRUCell(z_dim + n_actions + msg_dim, h_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 2 * z_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(h_dim + e_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 2 * z_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # 重参数化技巧：z = mu + eps * std，让梯度能穿过采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def prior(self, h: torch.Tensor):
        # h: (B, h_dim) -> mu, logvar: (B, z_dim)
        mu, logvar = self.prior_net(h).chunk(2, dim=-1)
        return mu, torch.clamp(logvar, -5.0, 2.0)

    def posterior(self, h: torch.Tensor, e: torch.Tensor):
        # h: (B, h_dim), e: (B, e_dim) -> mu, logvar: (B, z_dim)
        mu, logvar = self.post_net(torch.cat([h, e], dim=-1)).chunk(2, dim=-1)
        return mu, torch.clamp(logvar, -5.0, 2.0)


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        msg_dim: int = MSG_DIM,
        n_actions: int = N_ACTIONS,
        e_dim: int = 64,
        h_dim: int = 96,
        z_dim: int = 16,
        use_decoder: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.msg_dim = msg_dim
        self.n_actions = n_actions
        self.e_dim = e_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.use_decoder = use_decoder

        self.encoder = Encoder(obs_dim, e_dim)

        # 目标编码器：只做 EMA 更新，不参与反向传播
        self.target_encoder = Encoder(obs_dim, e_dim)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.rssm = RSSM(z_dim, h_dim, e_dim, n_actions, msg_dim)

        feat_dim = h_dim + z_dim
        self.reward_head = nn.Sequential(
            nn.Linear(feat_dim + n_actions, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.belief_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )
        # 价值头：预测从当前潜状态出发的折扣回报，给规划器提供“稠密”的方向感
        self.value_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.jepa_head = nn.Sequential(
            nn.Linear(feat_dim + n_actions, 128),
            nn.SiLU(),
            nn.Linear(128, e_dim),
        )
        self.decoder = (
            nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.SiLU(),
                nn.Linear(128, obs_dim),
            )
            if use_decoder
            else None
        )

    # ---------- 序列前向 ----------
    def encode_sequence(
        self,
        obs: torch.Tensor,
        msg: torch.Tensor,
        act: torch.Tensor,
        sample_z: bool = True,
    ) -> dict:
        """
        obs: (B, L, 150)
        msg: (B, L, 6)   当前步执行动作前的消息
        act: (B, L)      当前步执行的动作
        """
        B, L, _ = obs.shape
        e = self.encoder(obs.reshape(B * L, -1)).reshape(B, L, self.e_dim)

        h = torch.zeros(B, self.h_dim, device=obs.device)
        z = torch.zeros(B, self.z_dim, device=obs.device)
        zero_action = torch.zeros(B, self.n_actions, device=obs.device)

        hs, zs, priors, posts = [], [], [], []
        for t in range(L):
            # 第 t 步的 GRU 输入是“上一步的动作” a_{t-1}
            a_prev = zero_action if t == 0 else F.one_hot(
                act[:, t - 1], self.n_actions
            ).float()

            h = self.rssm.gru(torch.cat([z, a_prev, msg[:, t]], dim=-1), h)
            mu_p, logvar_p = self.rssm.prior(h)
            mu_q, logvar_q = self.rssm.posterior(h, e[:, t])
            z = (
                self.rssm.reparameterize(mu_q, logvar_q)
                if sample_z
                else mu_q
            )

            hs.append(h)
            zs.append(z)
            priors.append((mu_p, logvar_p))
            posts.append((mu_q, logvar_q))

        return {
            "e": e,
            "h": torch.stack(hs, dim=1),
            "z": torch.stack(zs, dim=1),
            "prior": (
                torch.stack([p[0] for p in priors], dim=1),
                torch.stack([p[1] for p in priors], dim=1),
            ),
            "posterior": (
                torch.stack([p[0] for p in posts], dim=1),
                torch.stack([p[1] for p in posts], dim=1),
            ),
        }

    # ---------- 单步在线推断（规划/评估用） ----------
    def observe_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        prev_action: torch.Tensor | None,
        msg_t: torch.Tensor,
        obs_t: torch.Tensor,
    ):
        """给定上一步状态、上一步动作、当前消息和当前观测，得到当前 (h,z)。"""
        B = obs_t.shape[0]
        if prev_action is None:
            a_prev = torch.zeros(B, self.n_actions, device=obs_t.device)
        else:
            a_prev = F.one_hot(prev_action, self.n_actions).float()

        h = self.rssm.gru(torch.cat([z_prev, a_prev, msg_t], dim=-1), h_prev)
        e = self.encoder(obs_t)
        mu_p, logvar_p = self.rssm.prior(h)
        mu_q, logvar_q = self.rssm.posterior(h, e)
        z = mu_q  # 评估时用均值，减少随机性
        return h, z, (mu_p, logvar_p), (mu_q, logvar_q)

    # ---------- 先验单步（规划用） ----------
    def prior_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
        msg: torch.Tensor,
    ):
        """在潜空间里前进一步：用 prior，不需要真实观测。"""
        a_onehot = F.one_hot(action, self.n_actions).float()
        h_next = self.rssm.gru(torch.cat([z, a_onehot, msg], dim=-1), h)
        mu_p, _ = self.rssm.prior(h_next)
        return h_next, mu_p  # 用均值，规划更稳定

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        # h: (B, h_dim), z: (B, z_dim), action: (B,)
        feat = torch.cat([h, z], dim=-1)
        a_onehot = F.one_hot(action, self.n_actions).float()
        return self.reward_head(torch.cat([feat, a_onehot], dim=-1)).squeeze(-1)

    # ---------- 目标编码器 EMA ----------
    @torch.no_grad()
    def update_target(self, decay: float = 0.99):
        for p_t, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_t.mul_(decay).add_(p, alpha=1.0 - decay)

    # ---------- 损失 ----------
    def compute_losses(
        self,
        batch: dict,
        free_bits: float = 0.5,
        kl_dyn_w: float = 0.8,
        kl_rep_w: float = 0.2,
        jepa_w: float = 0.5,
        belief_w: float = 1.0,
        value_w: float = 0.5,
        gamma: float = 0.99,
        kl_w: float = 0.1,
        var_w: float = 0.1,
        recon_w: float = 0.5,
    ):
        obs = batch["obs"]            # (B,L,150)
        msg = batch["msg"]            # (B,L,6)
        act = batch["act"]            # (B,L)
        rew = batch["rew"]            # (B,L)
        goal = batch["goal"]          # (B,L)
        valid = batch["valid"]        # (B,L)
        next_obs = batch["next_obs"]  # (B,L,150)
        valid_next = batch["valid_next"]

        out = self.encode_sequence(obs, msg, act, sample_z=True)
        h, z, e = out["h"], out["z"], out["e"]
        feat = torch.cat([h, z], dim=-1)          # (B,L,h_dim+z_dim)
        a_onehot = F.one_hot(act, self.n_actions).float()

        # 1) 奖励预测
        r_pred = self.reward_head(torch.cat([feat, a_onehot], dim=-1)).squeeze(-1)
        loss_reward = masked_mean(F.smooth_l1_loss(r_pred, rew, reduction="none"), valid)

        # 2) 隐藏门信念（训练时用特权标签 goal）
        logits = self.belief_head(feat)  # (B,L,3)
        ce = F.cross_entropy(
            logits.reshape(-1, 3), goal.reshape(-1), reduction="none"
        ).reshape_as(goal)
        loss_belief = masked_mean(ce, valid)

        # 2b) 价值：用真实回报（return-to-go）监督，给规划器提供稠密信号
        with torch.no_grad():
            returns = torch.zeros_like(rew)
            running = torch.zeros(rew.shape[0], device=rew.device)
            for t in range(rew.shape[1] - 1, -1, -1):
                running = rew[:, t] + gamma * running * (1.0 - batch["done"][:, t])
                returns[:, t] = running
                running = running * valid[:, t]   # 越过 episode 末尾后清零
        value_pred = self.value_head(feat).squeeze(-1)
        loss_value = masked_mean(
            F.smooth_l1_loss(value_pred, returns, reduction="none"), valid
        )

        # 3) JEPA：用 (h_t,z_t,a_t) 预测下一时刻的观测嵌入 e_{t+1}
        pred_e = self.jepa_head(
            torch.cat([feat[:, :-1], a_onehot[:, :-1]], dim=-1)
        )  # (B,L-1,e_dim)
        with torch.no_grad():
            target_e = self.target_encoder(
                next_obs[:, :-1].reshape(-1, self.obs_dim)
            ).reshape_as(pred_e)
        jepa_err = 1.0 - F.cosine_similarity(pred_e, target_e, dim=-1)
        loss_jepa = masked_mean(jepa_err, valid_next[:, :-1])

        # 4) KL：prior 与 posterior 的平衡 KL + free bits
        mu_p, logvar_p = out["prior"]
        mu_q, logvar_q = out["posterior"]
        kl_dyn = gaussian_kl(mu_q.detach(), logvar_q.detach(), mu_p, logvar_p)
        kl_rep = gaussian_kl(mu_q, logvar_q, mu_p.detach(), logvar_p.detach())
        # 监控用：裁剪前的原始 KL，判断 posterior 是否真的在用这些维度
        kl_raw = gaussian_kl(mu_q.detach(), logvar_q.detach(), mu_p, logvar_p)
        kl_dyn = torch.clamp(kl_dyn, min=free_bits)
        kl_rep = torch.clamp(kl_rep, min=free_bits)
        loss_kl = (
            kl_dyn_w * masked_mean(kl_dyn.sum(dim=-1), valid)
            + kl_rep_w * masked_mean(kl_rep.sum(dim=-1), valid)
        )

        # 5) 方差正则：防止所有嵌入缩成一个点
        e_valid = e[valid.bool()]
        if e_valid.shape[0] > 1:
            std = torch.sqrt(e_valid.var(dim=0) + 1e-4)
            loss_var = torch.relu(1.0 - std).mean()
        else:
            loss_var = torch.zeros((), device=obs.device)

        # 6) 可选解码器（1a 对照）
        loss_recon = torch.zeros((), device=obs.device)
        if self.decoder is not None:
            recon = self.decoder(feat)
            loss_recon = masked_mean(((recon - obs) ** 2).mean(dim=-1), valid)

        loss = (
            loss_reward
            + belief_w * loss_belief
            + value_w * loss_value
            + jepa_w * loss_jepa
            + kl_w * loss_kl
            + var_w * loss_var
            + recon_w * loss_recon
        )

        # 监控指标（不参与梯度）
        with torch.no_grad():
            mask = valid.bool()
            if mask.any():
                reward_mae = (r_pred[mask] - rew[mask]).abs().mean()
                belief_acc = (logits[mask].argmax(-1) == goal[mask]).float().mean()
                probs = torch.softmax(logits[mask], dim=-1)
                onehot = F.one_hot(goal[mask], 3).float()
                brier = ((probs - onehot) ** 2).sum(-1).mean()
                value_mae = (value_pred[mask] - returns[mask]).abs().mean()
                kl_total = kl_dyn.sum(-1)[mask].mean()
                kl_raw_total = kl_raw.sum(-1)[mask].mean()
                active_dims = (kl_raw > free_bits).float().sum(-1)[mask].mean()
                post_std = torch.exp(0.5 * logvar_q)[mask].mean()
                prior_std = torch.exp(0.5 * logvar_p)[mask].mean()
                z_std = z[mask].std(dim=0).mean()
            else:
                zero = torch.zeros((), device=obs.device)
                reward_mae = belief_acc = brier = kl_total = zero
                value_mae = kl_raw_total = active_dims = post_std = prior_std = z_std = zero

            jepa_mask = valid_next[:, :-1].bool()
            if jepa_mask.any():
                jepa_cos = F.cosine_similarity(pred_e, target_e, dim=-1)[jepa_mask].mean()
            else:
                jepa_cos = torch.zeros((), device=obs.device)

        metrics = {
            "loss": loss.detach(),
            "loss_reward": loss_reward.detach(),
            "loss_belief": loss_belief.detach(),
            "loss_value": loss_value.detach(),
            "loss_jepa": loss_jepa.detach(),
            "loss_kl": loss_kl.detach(),
            "loss_var": loss_var.detach(),
            "loss_recon": loss_recon.detach(),
            "reward_mae": reward_mae,
            "value_mae": value_mae,
            "belief_acc": belief_acc,
            "brier": brier,
            "jepa_cos": jepa_cos,
            "kl": kl_total,
            "kl_raw": kl_raw_total,
            "active_dims": active_dims,
            "post_std": post_std,
            "prior_std": prior_std,
            "z_std": z_std,
        }
        return loss, metrics


# ---------- 工具函数 ----------
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对 mask=1 的位置求平均。mask 的 shape 和 x 一样。"""
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    """逐维 KL(q || p)，两个都是对角高斯。logvar 是 log(方差)。"""
    return 0.5 * (
        logvar_p
        - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
        - 1.0
    )
