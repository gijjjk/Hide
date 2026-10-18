"""
教程 3：用 PyTorch 的标准训练循环训练一个小 MLP。

运行：
    python tutorials/03_train_mlp.py

目标：看懂 nn.Module、optimizer、loss.backward()、optimizer.step() 的配合。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn

torch.manual_seed(0)

# ---------- 1. 造一个玩具数据集 ----------
# 目标函数：y = sin(3x) + 0.1 * 噪声
x = torch.linspace(-2.0, 2.0, 256).unsqueeze(1)   # shape: (256, 1)
y = torch.sin(3 * x) + 0.1 * torch.randn_like(x)   # shape: (256, 1)

# ---------- 2. 定义模型 ----------
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.SiLU(),
    nn.Linear(32, 32),
    nn.SiLU(),
    nn.Linear(32, 1),
)
print("模型结构：\n", model)

# ---------- 3. 损失函数和优化器 ----------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# ---------- 4. 标准训练循环 ----------
for step in range(501):
    pred = model(x)                 # 前向：输入 x，得到预测
    loss = loss_fn(pred, y)         # 算误差

    optimizer.zero_grad()           # 清空上一次的梯度
    loss.backward()                 # 反向传播：算每个参数的梯度
    optimizer.step()                # 用梯度更新参数

    if step % 100 == 0:
        print(f"step={step:3d}  loss={loss.item():.4f}")

# ---------- 5. 看看训练后的预测 ----------
with torch.no_grad():
    test_x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    test_pred = model(test_x)
    test_true = torch.sin(3 * test_x)
    print("\n输入      预测      真实值")
    for xi, pi, ti in zip(test_x.flatten(), test_pred.flatten(), test_true.flatten()):
        print(f"{xi.item():+.2f}    {pi.item():+.3f}    {ti.item():+.3f}")

print("\n如果 loss 从大变小，说明训练循环工作正常。")
