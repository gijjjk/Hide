"""
教程 2：张量、shape、autograd（自动求梯度）。

运行：
    python tutorials/02_tensor_autograd.py

你需要先安装 CPU 版 PyTorch：
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import torch

print("PyTorch 版本：", torch.__version__)

# ---------- 1. 张量就是“带形状的数组” ----------
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.ones(2, 2)
print("\n1. 张量：")
print("a =\n", a)
print("a.shape =", a.shape)
print("a + b =\n", a + b)

# ---------- 2. 自动求梯度 ----------
# requires_grad=True 表示“请帮我记录对这个张量的运算，之后要算梯度”
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()          # y = 1^2 + 2^2 + 3^2
y.backward()                # 反向传播，计算 dy/dx
print("\n2. autograd：")
print("y =", y.item())
print("dy/dx =", x.grad)    # 应该是 2x = [2, 4, 6]

# ---------- 3. 用梯度下降拟合 y = 2x + 1 ----------
print("\n3. 手动训练一个线性模型：")
xs = torch.linspace(-1, 1, 100)
ys = 2 * xs + 1

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(201):
    pred = w * xs + b
    loss = ((pred - ys) ** 2).mean()

    loss.backward()

    # 更新参数时不要记录梯度，所以用 torch.no_grad()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if step % 50 == 0:
        print(f"step={step:3d}  loss={loss.item():.6f}  w={w.item():.3f}  b={b.item():.3f}")

print("\n训练结束后，w 应该接近 2，b 应该接近 1。")
print("这就是所有神经网络训练的核心循环：预测 → 算 loss → backward → 更新参数。")
