# AskDreamer：一个 CPU 可训练的“潜状态—思考—询问”小架构

> 一个面向高中生的研究原型：**不重建像素，只在潜空间里预测和规划；不确定时主动花代价询问。**
>
> 它不是大语言模型，也不是 DreamerV3 的复现。它借用 DreamerV3 的 RSSM/潜空间规划思想，把 **ACT / THINK / ASK** 三个动作放进同一个回路，并在一个小网格世界里做可复现的消融实验。

---

## 1. 这个项目想验证什么

| 编号 | 假设 | 对照 | 结论（详见报告） |
|---|---|---|---|
| **H1** | 不重建观测、只预测下一时刻潜状态，也足够支撑任务 | 加解码器重建 150 维观测 | **部分成立**：无解码器小 16%，任务指标有竞争力，但没有全面胜出 |
| **H2** | 在潜空间里“多想几条”比反应式动作更好 | 采样数 N=1/8/32/128，视野 H=4/8/12 | **成立**：N 从 1 到 128，成功率 10% → 97.8%；H=4 的回报/延迟最好 |
| **H3** | 用不确定性/信息价值决定“何时问”比固定问更好 | 从不问 / 固定问 / 熵门控 / VOI 门控 | **部分成立**：VOI 能随询问成本自动改变询问率，但不是每个成本下都最高 |

完整数据在 [`report/tables.md`](report/tables.md)，完整分析在 [`report/report.md`](report/report.md)。

---

## 2. 环境长什么样

`HiddenGoalGrid` 是一个 7×7 的网格：

- 智能体从 `(1,1)` 出发，三扇门固定在右侧；
- 每局随机一扇门是正确门 `goal ∈ {0,1,2}`，**goal 不会进入观测**；
- 智能体只能看到以自己为中心的 **5×5 局部视野**；
- 观测是 **5×5×6 = 150 维**向量（墙、门 0/1/2、告示牌、自己），不是图片；
- 另有一条独立的 **6 维消息通道**：前 3 维是告示牌排除的门，后 3 维是 oracle 的回答；
- 动作 5 个：上/下/左/右/**ASK**（启用 ABSTAIN 时额外增加第 6 个动作，默认关闭）；
- 正确门 +1；错误门 −0.3 并变成墙；每步 −0.01；ASK 额外 −0.05；40 步超时再 −0.1；
- 告示牌会排除一扇错误门，所以“问不问”不总是显然的。

**为什么把答案放在独立消息通道？**
如果 oracle 答案直接拼进观测，信念头可以“抄答案”，H3 就失去意义。现在观测里没有 goal，答案只能通过 `ASK` 的消息通道进入，智能体必须自己决定要不要买这个信息。

---

## 3. 架构（比 DreamerV3 小很多）

```
观测 o_t (150) ──► Encoder ──► e_t (64)
                                  │
消息 m_t (6) ─────────────┐       │
动作 a_{t-1} (5/6) ───────┼──► GRU ──► h_t (96)
潜变量 z_{t-1} (16) ──────┘             │
                                        ├── prior   p(z_t | h_t)
                                        ├── posterior q(z_t | h_t, e_t)   （训练时）
                                        │
                                        ├── 奖励头 R(h,z,a)
                                        ├── 信念头 B(h,z) → 3 类“哪扇门对”
                                        ├── 价值头 V(h,z) → 折扣回报
                                        └── JEPA 头 → 预测下一时刻 e_{t+1}
```

- **RSSM-lite**：GRU 维护确定性状态 `h_t`，`z_t` 是 16 维高斯潜变量；训练用 posterior，想象/规划用 prior。
- **不重建像素**：主模型没有解码器，JEPA 头用 stop-gradient + EMA target encoder 预测下一时刻的嵌入；VICReg 式方差项防止嵌入缩成一个点。
- **可选的解码器**：默认带解码器；`--no-decoder` 关闭。它只作为 H1 的对照，不作为主目标。
- **思考**：随机 shooting。从当前 `(h,z)` 出发，采样 N 条长度 H 的移动动作序列，用 prior rollout，累加预测奖励，最后加 `γ^H · V(h_H,z_H)`；执行得分最高序列的第一个动作。N 就是“思考预算”。
- **询问**：`ASK` 由单独的控制器决定，不放进随机 shooting 的动作空间。控制器有三种：
  - `fixed`：第一次有机会就问；
  - `entropy`：信念熵 > 阈值才问；
  - `voi`：比较“不问的期望试错成本”和“询问成本”，只有前者更大才问；
  - `sign_first`：先走到告示牌，再决定问不问（一个规则式探索先验）。

参数量：**173,029（无解码器） / 206,843（有解码器）**，在 4 核 CPU 上训练 2000 步约 **93–97 秒**。

---

## 4. 安装（Windows + CPU）

在 `D:\hide` 下打开 PowerShell 或 Git Bash：

```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell
# source .venv/bin/activate     # Git Bash

python -m pip install --upgrade pip

# 关键：安装 CPU 版 PyTorch，不要装 CUDA 版
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m pip install numpy matplotlib
```

也可以：

```bash
python -m pip install -r requirements.txt
```

> `requirements.txt` 里的 `torch` 默认会从 PyPI 拉取；如果你的网络/平台装到了 CUDA 版，请用上面的 `--index-url` 重新安装。你的 GeForce MX110（CUDA 10.1）**不需要也不应该**参与训练，项目全程用 CPU。

检查安装：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

应看到类似 `2.x.x+cpu False`。

---

## 5. 运行顺序（从零到结果）

### 5.1 先补最小基础（可选，但推荐）

```bash
python tutorials/01_python_basics.py
python tutorials/02_tensor_autograd.py
python tutorials/03_train_mlp.py
```

这三个脚本分别演示：列表/字典/类、张量与 autograd、一个最小 PyTorch 训练循环。

### 5.2 环境和随机基线

```bash
python tests/test_env.py
python steps/01_env.py --episodes 1 --render
python steps/01_env.py --episodes 200
```

200 局随机策略的参考结果：成功率约 **7.5%**，平均回报约 **−0.84**。随机策略偶尔能撞对门，但不会主动询问。

### 5.3 训练世界模型

冒烟测试（几十秒，只验证流程）：

```bash
python steps/02_world_model.py --smoke
```

正式训练无解码器版本（主模型）：

```bash
python steps/02_world_model.py --episodes 400 --steps 2000 --no-decoder \
  --out experiments/checkpoints/world_model_nodec.pt \
  --metrics-out experiments/logs/world_model_nodec_metrics.json
```

训练有解码器版本（H1 对照）：

```bash
python steps/02_world_model.py --episodes 400 --steps 2000 \
  --out experiments/checkpoints/world_model_dec.pt \
  --metrics-out experiments/logs/world_model_dec_metrics.json
```

### 5.4 在潜空间里规划（H2）

```bash
python steps/03_plan.py --episodes 100 --samples 32 --horizon 4 --ask fixed \
  --checkpoint experiments/checkpoints/world_model_nodec.pt
```

对比思考预算：

```bash
python steps/03_plan.py --samples 1   --horizon 8 --ask fixed --checkpoint experiments/checkpoints/world_model_nodec.pt
python steps/03_plan.py --samples 8   --horizon 8 --ask fixed --checkpoint experiments/checkpoints/world_model_nodec.pt
python steps/03_plan.py --samples 32  --horizon 8 --ask fixed --checkpoint experiments/checkpoints/world_model_nodec.pt
python steps/03_plan.py --samples 128 --horizon 8 --ask fixed --checkpoint experiments/checkpoints/world_model_nodec.pt
```

### 5.5 对比询问策略（H3）

一条命令跑 `never / fixed / entropy / voi`：

```bash
python steps/04_ask.py --policy all --sign-first --ask-cost -0.05 \
  --episodes 50 --seed 0 --checkpoint experiments/checkpoints/world_model_nodec.pt
```

单独跑一个策略：

```bash
python steps/04_ask.py --policy voi --sign-first --ask-cost -0.4 \
  --episodes 50 --seed 0
```

### 5.6 复现报告里的 3-seed 表

```bash
# 1) 先按 report/report.md 第 12 节训练 6 个 checkpoint
#    （3 seeds × 无解码器/有解码器；run_matrix.py 只做评估，不训练模型）
# 2) 跑 H2 + H3 矩阵（45 个评估任务，约 10–20 分钟）
python experiments/run_matrix.py
# 3) 汇总表格和图
python experiments/summarize.py
```

输出：

- `report/tables.md`
- `report/figures/h1_compare.png`
- `report/figures/h2_budget.png`
- `report/figures/h3_ask.png`

> `run_matrix.py` 默认使用 `world_model_nodec.pt`、`world_model_nodec_s1.pt`、`world_model_nodec_s2.pt` 三个 checkpoint。seed 0 的 checkpoint 没有 `_s0` 后缀。

### 5.7 分布外测试（7×7 训练 → 10×10 测试）

同一个 checkpoint，不改模型、不重新训练，只把环境换成 10×10：

```bash
python experiments/eval_ood.py \
  --checkpoint experiments/checkpoints/world_model_nodec.pt \
  --train-grid 7 --test-grid 10 \
  --episodes 50 --samples 32 --horizon 8 --ask never --seed 0
```

输出：

- `experiments/logs/ood_7to10_seed0.json`（train/test 汇总 + 逐局记录）
- `report/figures/ood_compare.png`（2×3 对比图）

`--ask never` 让信念只能从观测/历史推断，用来单独看校准和规划。要看询问行为时换成：

```bash
python experiments/eval_ood.py --ask entropy \
  --out experiments/logs/ood_7to10_seed0_entropy.json
```

该设置下 7×7/10×10 成功率为 0.92/0.72，询问率都是 1.0（熵门控在第一步就询问），
说明“主动获取信息”能缓解一部分掉点，但不能消除布局变化带来的规划困难。

已跑结果（seed 0，50 局，无解码器 checkpoint）：

| 指标 | 7×7 | 10×10 | 变化 |
|---|---:|---:|---:|
| 成功率 | 0.740 | 0.480 | -0.260 |
| 平均回报 | +0.245 | -0.076 | -0.321 |
| 平均步数 | 26.5 | 33.0 | +6.5 |
| 信念准确率 | 0.444 | 0.445 | +0.001 |
| Brier（越低越好） | 0.714 | 0.748 | +0.034 |
| 询问率 | 0.000 | 0.000 | 0 |

预期现象：10×10 下成功率、回报和步数明显变差，Brier 上升；信念准确率不一定下降，
但置信度会更不校准（Brier 比 accuracy 更敏感）。5×5 视野在 10×10 上覆盖比例更低，
门和边界的绝对位置也没在训练中出现过，掉点是预期的。

已知局限：

- 同一 checkpoint 直接迁移，没有在 10×10 上重训；
- 门数固定为 3，只改变网格尺寸和门/告示牌位置；
- `max_steps=40` 和 `horizon=8` 没有随网格缩放；`--horizon 12` 的对照没有恢复成功率
  （10×10 下 0.360，反而低于 `horizon=8` 的 0.480），说明主要不是规划视野不足；
- 同一 seed 下 10×10 的告示牌候选集合不同，可读信息分布也不同；
- 50 局样本较小，建议用 `--seed 1`、`--seed 2` 重复；
- `world_model.pt` 是早期权重（没有 value_head，belief_acc≈0），不建议用于对比；
  主结果使用 `world_model_nodec.pt`/`world_model_dec.pt`。

快速冒烟：

```bash
python experiments/eval_ood.py --episodes 3 --samples 8 --horizon 4 \
  --out experiments/logs/_ood_smoke.json --figure report/figures/_ood_smoke.png
```

### 5.8 ABSTAIN 动作（任务 2）

ABSTAIN 是可选的第 6 个动作（`enable_abstain=True`，动作编号 5）：

- 立即结束本局，`done=True`；
- 不拿成功奖励，也不触发错门罚分；
- 只付 `step_penalty + abstain_penalty`（默认 `-0.01 + -0.1`）。

默认关闭，此时动作空间仍是 5，旧 checkpoint 和旧命令完全兼容。
启用后动作空间为 6，**旧的 5 动作 checkpoint 不能用于 6 动作环境**，反之亦然；
`steps/03_plan.py` 和 `experiments/eval_ood.py` 会检查 `config["n_actions"]` 并在不匹配时退出。

训练 6 动作世界模型（不会覆盖旧的 `world_model.pt` / `world_model_nodec.pt`）：

```bash
python steps/02_world_model.py --enable-abstain --episodes 400 --steps 2000 \
  --no-decoder --seed 0 --abstain-prob 0.05 --random-abstain-prob 0.02
```

输出：

- `experiments/checkpoints/world_model_abstain.pt`
- `experiments/logs/world_model_abstain_metrics.json`

评估三种弃权门控（`never / entropy / voi`）：

```bash
python steps/03_plan.py --episodes 100 --samples 32 --horizon 8 --ask never \
  --enable-abstain --abstain entropy --abstain-threshold 1.0 \
  --checkpoint experiments/checkpoints/world_model_abstain.pt \
  --out experiments/logs/abstain_entropy_t10.json
```

新增指标：

- `abstain_rate`：弃权局数 / 总局数；
- `coverage`：未弃权局数 / 总局数；
- `selective_success`：未弃权局中的成功率。

seed 0，100 局，7×7，无解码器：

| 配置 | 成功率 | 弃权率 | 覆盖率 | 覆盖率内成功率 |
|---|---:|---:|---:|---:|
| 5 动作基线（`world_model_nodec.pt`） | 0.700 | 0.000 | 1.000 | 0.700 |
| 6 动作，`--abstain never` | 0.480 | 0.000 | 1.000 | 0.480 |
| 6 动作，熵门控 τ=0.5 | 0.080 | 0.680 | 0.320 | 0.250 |
| 6 动作，熵门控 τ=0.8 | 0.300 | 0.260 | 0.740 | 0.405 |
| 6 动作，熵门控 τ=1.0 | 0.410 | 0.070 | 0.930 | 0.441 |
| 6 动作，VOI 门控（margin=0） | 0.140 | 0.590 | 0.410 | 0.341 |
| 6 动作，VOI 门控（margin=0.2） | 0.400 | 0.040 | 0.960 | 0.417 |
| 6 动作，VOI 门控（margin=0.5） | 0.480 | 0.000 | 1.000 | 0.480 |

seed 1 的 `--abstain never` 成功率为 0.460，与 seed 0 的 0.480 接近。

结论与局限：

- 机制本身可用：ABSTAIN 能正确终止回合、改变动作空间，训练和评估都能区分
  `abstain_rate / coverage / selective_success`。
- **规则门控没有带来选择性收益**：所有会弃权的配置，其覆盖率内成功率都低于
  “从不弃权”的 0.480；阈值越松，弃权越多，总成功率越低。
- 6 动作模型本身比 5 动作基线弱（0.480 vs 0.700）。同预算下重新初始化会改变
  参数随机流；即使使用完全相同的 5 动作数据训练 6 动作模型，成功率也只有 0.47 左右。
  因此 6 动作与 5 动作之间的差距不能全部归因于 ABSTAIN。
- 为了让 6 动作数据保留足够多的成功样本，训练时把弃权概率压得很低
  （脚本 0.05，随机 0.02）。ABSTAIN 转移很少，世界模型对动作 5 的奖励预测不可靠；
  当前的弃权门控都是规则式，不依赖该预测。

分布外结果（同一个 6 动作 checkpoint，7×7 → 10×10，50 局）：

| 配置 | 网格 | 成功率 | Brier | 信念熵 | 弃权率 |
|---|---|---:|---:|---:|---:|
| `never` | 7×7 | 0.460 | 0.905 | 0.463 | 0.000 |
| `never` | 10×10 | 0.260 | 1.012 | 0.453 | 0.000 |
| 熵门控 τ=1.0 | 7×7 | 0.340 | 0.897 | 0.474 | 0.120 |
| 熵门控 τ=1.0 | 10×10 | 0.140 | 1.084 | 0.423 | 0.020 |

分布外时 Brier 明显变差、成功率下降，但**信念熵没有上升，弃权率反而下降**。
也就是说，信念头在分布外更多是“自信地错”，而不是“知道自己不知道”；
基于信念熵的 ABSTAIN 门控因此无法检测分布外。这是任务 2 的主要负结果。

输出文件：

- `experiments/logs/abstain_*.json`
- `experiments/logs/ood_7to10_seed0_abstain_never.json`
- `experiments/logs/ood_7to10_seed0_abstain_entropy.json`
- `report/figures/ood_abstain_never_compare.png`
- `report/figures/ood_abstain_compare.png`

---

## 6. 目录结构

```
D:\hide\
  README.md                  # 本文件
  requirements.txt
  askdreamer/                # 正式包
    env.py                   # HiddenGoalGrid（纯 Python + numpy）
    data.py                  # 脚本策略、数据收集、序列采样
    model.py                 # Encoder + RSSM + 奖励/信念/价值/JEPA 头
    train.py                 # 训练、评估、线性探针、有效秩
    planner.py               # 随机 shooting 潜空间规划器
    controller.py            # never / fixed / entropy / VOI / sign-first
  tutorials/                 # 给 Python 不熟的人的最小基础
    01_python_basics.py
    02_tensor_autograd.py
    03_train_mlp.py
  steps/                     # 单文件、可独立运行的教学版
    01_env.py
    02_world_model.py
    03_plan.py
    04_ask.py
  experiments/
    run_matrix.py            # 批量跑 3 seeds × H2/H3
    summarize.py             # 从 JSON 生成表格和图
    checkpoints/             # .pt 模型
    logs/                    # 每个配置一个 JSON
  tests/
    test_env.py              # 环境自检（9 项）
    test_env_grid.py         # 网格尺寸参数化回归测试
    test_abstain.py          # ABSTAIN 动作与弃权门控测试
    test_shapes.py           # 张量形状/前向/反向/规划器/控制器自检
    test_overfit.py          # 小 batch 过拟合，确认模型能学
  report/
    report.md                # 中文实验报告
    tables.md                # 自动生成的完整表格
    figures/                 # 三张结果图
```

---

## 7. 主要结果（3 seeds，均值 ± 样本标准差）

### H1：无解码器 vs 有解码器

| 版本 | 参数量 | 训练秒 | reward MAE | 信念准确率 | Brier | 线性探针 | 有效秩 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 有解码器 | 206,843 | 97.1 ± 2.0 | 0.077 ± 0.006 | 0.934 ± 0.004 | 0.077 ± 0.003 | 0.924 ± 0.022 | 8.50 ± 0.25 |
| 无解码器 | 173,029 | 92.7 ± 2.1 | 0.076 ± 0.008 | 0.932 ± 0.003 | 0.080 ± 0.007 | 0.928 ± 0.011 | 8.34 ± 0.28 |

**读法**：无解码器小 16%，reward MAE 和线性探针略好；Brier/JEPA/有效秩略差。结论是“有竞争力”，不是“全面胜出”。

### H2：思考预算

| 配置 | 成功率 | 平均回报 | 平均步数 | 单步规划 ms |
|---|---:|---:|---:|---:|
| N=1, H=8 | 0.100 ± 0.058 | −0.496 ± 0.070 | 38.6 ± 1.4 | 3.61 ± 0.31 |
| N=8, H=8 | 0.567 ± 0.219 | 0.106 ± 0.270 | 30.4 ± 3.8 | 4.52 ± 0.18 |
| N=32, H=8 | 0.878 ± 0.126 | 0.491 ± 0.143 | 22.2 ± 2.4 | 5.08 ± 0.19 |
| N=128, H=8 | 0.978 ± 0.038 | 0.674 ± 0.038 | 15.8 ± 1.3 | 7.79 ± 0.39 |
| N=32, H=4 | 0.911 ± 0.096 | 0.626 ± 0.138 | 18.6 ± 4.0 | 2.62 ± 0.25 |
| N=32, H=12 | 0.911 ± 0.096 | 0.444 ± 0.121 | 24.5 ± 1.0 | 7.80 ± 0.57 |

**读法**：N 越大越好；但 H 不是越大越好，H=4 比 H=8/12 回报更高、延迟更低。prior 长 rollout 会漂移。

### H3：询问策略

| 策略 | 询问成本 | 成功率 | 平均回报 | 询问率 |
|---|---:|---:|---:|---:|
| 先读牌 + 从不问 | 0.00 | 0.807 ± 0.064 | 0.296 ± 0.061 | 0.000 |
| 先读牌 + 固定问 | −0.05 | 0.953 ± 0.042 | 0.592 ± 0.063 | 1.000 |
| 先读牌 + 固定问 | −0.40 | 0.953 ± 0.042 | 0.242 ± 0.063 | 1.000 |
| 先读牌 + 熵门控 τ=0.8 | −0.05 | 0.873 ± 0.064 | 0.470 ± 0.118 | 0.413 |
| 先读牌 + 熵门控 τ=0.8 | −0.40 | 0.873 ± 0.064 | 0.325 ± 0.114 | 0.413 |
| 先读牌 + VOI 门控 | −0.05 | 0.953 ± 0.042 | 0.590 ± 0.060 | 0.967 |
| 先读牌 + VOI 门控 | −0.40 | 0.807 ± 0.064 | 0.296 ± 0.061 | 0.000 |
| 不先读牌 + 固定问 | −0.05 | 0.867 ± 0.129 | 0.499 ± 0.150 | 1.000 |
| 不先读牌 + VOI 门控 | −0.40 | 0.827 ± 0.092 | 0.121 ± 0.128 | 0.913 |

**读法**：低询问成本下，固定问和 VOI 都接近最好；高询问成本下，固定问最差，VOI 自动退化为“不问”。VOI 的价值是**跨成本自适应**，不是每个 regime 都最高。

---

## 8. 给 Python 不熟的人：怎么读代码

建议顺序：

1. `tutorials/01_python_basics.py` → 知道 `list`、`dict`、`class` 是什么；
2. `askdreamer/env.py` → 环境逻辑，重点看 `observe()` 和 `step()`；
3. `steps/01_env.py` → 随机策略基线；
4. `tutorials/02_tensor_autograd.py` → 张量、`backward()`；
5. `tutorials/03_train_mlp.py` → 最小训练循环；
6. `askdreamer/model.py` → 每个张量的 shape 都写在注释里；
7. `steps/02_world_model.py` → 数据收集 → 训练 → 评估；
8. `askdreamer/planner.py` → “在潜空间里想”其实就是一个 for 循环；
9. `askdreamer/controller.py` → 询问规则，最直观的“主动询问”；
10. `steps/03_plan.py` / `steps/04_ask.py` → 把前面拼起来。

**调试口诀**：看到维度错误，先打印 `shape`；先查 batch 维，再查特征维；一次只改一个变量。

---

## 9. 已知局限

- 这是一个**很小的网格世界**，结论不能直接推广到大模型或真实机器人。
- 观测是 150 维向量，不是像素；“不关注每个像素”在这里体现为“不重建观测”，但还没有做图像实验。
- 行为数据主要来自脚本策略 + 随机策略，不是完全自主学习。
- 信念头训练时用了 `goal` 特权标签，测试时只能靠历史/消息；报告里明确写了这一点。
- 询问控制器是**规则版**（熵/VOI），不是学习出来的；`sign_first` 也是规则式探索先验。
- ABSTAIN 门控同样是规则版；在 7×7 和 10×10 上，信念熵都没有区分“会成功”和“会失败”的局，
  弃权反而降低了总成功率（见 5.8）。
- 世界模型在 `ask_cost=-0.05` 的数据上训练，H3 的高成本（−0.40）评估存在奖励/价值尺度偏移；规划动作排序不受常数偏移影响，但回报绝对值要谨慎比较。
- 每个配置只有 3 个 seed、30–50 局，方差不小；结论看趋势，不要过度解读小数点。
- 没有训练 actor-critic / 策略蒸馏；没有语言输入输出。

---

## 10. 下一步可以做什么

按优先级：

1. **在多个询问成本下重新训练世界模型**，消除 H3 高成本评估的尺度偏移；
2. **学习型询问门控**：把 `ASK` 放进动作空间，或用 VOI 头做辅助监督；
3. **策略蒸馏**：用规划器的动作做行为克隆，比较“快系统 / 慢系统 / 快慢切换”；
4. **自适应思考预算**：top-1 与 top-2 得分差小于阈值时，把 N 从 32 增到 128；
5. **更多 seed 和更大网格**，检查结论稳健性；
6. **把观测换成图像**，验证 decoder-free 在真正“像素”场景下是否仍然有竞争力；
7. **把消息通道换成自然语言**，这是通向“会问问题的小语言模型”的一步（但那是另一个大项目）。

---

## 11. 参考

- DreamerV3 官方实现（JAX）：<https://github.com/danijar/dreamerv3>
- 从零实现 DreamerV3 的教学版（PyTorch）：<https://github.com/DrunkJin/dreamer-from-scratch>
- DreamerV3 PyTorch 移植：<https://github.com/NM512/dreamerv3-torch>
- JEPA 概念综述：<https://huggingface.co/spaces/mishig/jepawiki>
- 带代价的询问 / VOI：<https://www.ias.tu-darmstadt.de/uploads/Site/EditPublication/Hoelscher-msc-2017.pdf>
- 主动信息获取与信息价值（VoI）综述：<https://zenodo.org/records/23085343>
- 中文世界模型教材（JEPA 章节）：<https://walkinglabs.github.io/hands-on-world-models/chapters/06-jepa/01-jepa-foundation.html>

---

## 12. 一句话总结

> 在一个 7×7 的小世界里，我验证了：**不重建观测、只在潜空间里预测和规划是可行的；多“想”几条能显著提高成功率；而“什么时候问”应该由信息价值与询问成本的比较来决定，而不是固定地问或只看熵。**

负结果和局限同样写在报告里。这个项目不是要打败大模型，而是把“潜状态 + 思考 + 询问”三个机制跑通，并学会用对照实验判断它们是否真的有用。
