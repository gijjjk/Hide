"""
教程 1：最小 Python 基础。
运行：
    python tutorials/01_python_basics.py

目标：能看懂变量、列表、字典、循环、函数、类，以及 numpy 的 shape。
不需要 PyTorch。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

# 1. 变量和打印
name = "AskDreamer"
year = 2026
print("1. 变量：", name, year)

# 2. 列表：一串有顺序的东西
actions = ["up", "down", "left", "right", "ASK"]
print("2. 列表：", actions)
print("   第一个：", actions[0], " 最后一个：", actions[-1])

# 3. 字典：用键找值
config = {"lr": 3e-4, "batch": 32, "z_dim": 16}
print("3. 字典：", config, " 学习率 =", config["lr"])

# 4. for 循环和 if 判断
print("4. 循环：")
for index, action in enumerate(actions):
    if action == "ASK":
        print(f"   第 {index} 个动作是询问，需要花代价")
    else:
        print(f"   第 {index} 个动作是移动：{action}")

# 5. 函数：把一段逻辑装起来，重复使用
def discounted_return(rewards, gamma=0.9):
    """计算折扣回报：G = r0 + gamma*r1 + gamma^2*r2 + ..."""
    total = 0.0
    for t, r in enumerate(rewards):
        total += (gamma ** t) * r
    return total

print("5. 函数：折扣回报 =", discounted_return([1.0, 0.0, 0.0]))

# 6. 类：把数据和操作放在一起
class Counter:
    def __init__(self):
        self.value = 0

    def add(self, n=1):
        self.value += n

    def show(self):
        return f"Counter(value={self.value})"

c = Counter()
c.add()
c.add(5)
print("6. 类：", c.show())

# 7. numpy 数组和 shape（以后调试模型最常用的信息）
x = np.zeros((2, 3))
y = np.ones((3, 4))
z = x @ y
print("7. numpy：")
print("   x.shape =", x.shape, " y.shape =", y.shape)
print("   x @ y 的 shape =", z.shape, "（2x3 乘 3x4 得到 2x4）")

print("\n如果这些都能看懂，就可以进入教程 2。")
