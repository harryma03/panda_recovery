<div align="center">
  <h1>Panda Recovery</h1>
  <p>基于 Isaac Gym 与深度强化学习的 Panda 四足机器人摔倒自恢复项目</p>
</div>

---

## 📌 项目简介

本项目面向 Panda 四足机器人在摔倒后的自主恢复问题。策略从四脚朝天、左侧倒地、右侧倒地、俯仰翻转以及随机关节姿态等初始状态出发，学习翻身、支撑、起立并保持稳定站姿。

项目基于 Isaac Gym 进行大规模并行训练，使用 `ActorCriticBarlowTwins` 策略网络与 `NP3O` 约束强化学习算法。历史观测编码器通过 Barlow Twins 风格的自监督目标学习时序特征；恢复策略同时考虑机身姿态、站立高度、足端接触、关节姿态和恢复后的稳定性。

### 任务总览

| task 名 | 机器人 | 功能 | 训练入口 | 评估入口 |
|---|---|---|---|---|
| `pandaN3poHim` | Panda 四足机器人 | 多姿态摔倒恢复与稳定站立 | `train.py` | `simple_play.py` |

当前恢复测试覆盖：

- 精确四脚朝天与带倾角的四脚朝天
- 左侧倒地与右侧倒地
- 前后俯仰翻转
- 不同初始关节姿态
- 恢复后的站立稳定性

---

## 🔁 工作流程

```text
Train (Isaac Gym) → Play / Evaluation → ONNX Export → Sim2Sim → Sim2Real
```

- **Train**：在 Isaac Gym 中使用 4096 个并行环境训练恢复策略。
- **Play / Evaluation**：通过九宫格测试集检查不同摔倒姿态下的恢复效果。
- **ONNX Export**：Play 加载 checkpoint 后自动导出 `model.onnx`。
- **Sim2Sim**：将 ONNX 策略部署到 `rl_sar` 的 Gazebo 或 MuJoCo 环境验证。
- **Sim2Real**：完成关节顺序、控制周期、PD 参数和限幅检查后再部署至实机。

---

## 🛠️ 环境安装

### 1. 基础环境

推荐环境：

- Ubuntu 22.04
- Python 3.8
- NVIDIA GPU 与兼容的 CUDA 驱动
- NVIDIA Isaac Gym Preview 4

先创建并进入 Python 环境：

```bash
conda create -n rl python=3.8
conda activate rl
```

### 2. 安装 Isaac Gym

下载并解压 Isaac Gym 后，在其 `python` 目录执行：

```bash
cd ~/isaacgym/python
pip install -e .
```

可先运行 Isaac Gym 示例确认安装正常：

```bash
python examples/joint_monkey.py
```

### 3. 安装 Python 依赖

回到本项目根目录后安装运行所需依赖：

```bash
cd /path/to/panda_recovery
pip install numpy opencv-python pillow matplotlib tensorboard torchvision
```

PyTorch 版本需要与本机 CUDA 和 Isaac Gym 兼容，请优先使用当前 Isaac Gym 环境中已验证的版本，避免安装依赖时意外覆盖。

> 本项目通过根目录下的 Python 包直接运行，无需执行 `pip install -e .`。所有训练和评估命令都应在 `panda_recovery/` 根目录执行。

---

## 🚀 训练

### 启动训练

```bash
conda activate rl
cd /path/to/panda_recovery
python train.py --task pandaN3poHim --headless
```

训练时默认关闭图形界面，以提高并行仿真速度。需要临时观察环境时可去掉 `--headless`。

### 常用参数

| 参数 | 说明 |
|---|---|
| `--task` | 任务名称，本项目使用 `pandaN3poHim` |
| `--headless` | 关闭图形界面 |
| `--num_envs` | 覆盖并行环境数量 |
| `--max_iterations` | 覆盖最大训练迭代数 |
| `--seed` | 随机种子 |
| `--sim_device` | Isaac Gym 仿真设备，例如 `cuda:0` 或 `cpu` |
| `--rl_device` | 强化学习设备，例如 `cuda:0` |

例如，使用较少环境进行调试：

```bash
python train.py \
  --task pandaN3poHim \
  --num_envs 64 \
  --max_iterations 20
```

### 从 checkpoint 继续训练

当前训练 runner 读取 [`configs/panda_config.py`](configs/panda_config.py) 中的以下字段：

```python
class runner:
    resume = True
    resume_path = "logs/<experiment>/<run>/model_<iteration>.pt"
```

将 `resume_path` 修改为需要加载的模型，然后启动训练。若要完全从头训练，请设置：

```python
resume = False
```

> `--load_run` 和 `--checkpoint` 目前主要用于 `simple_play.py` 选择评估模型；训练 runner 仍以 `resume_path` 为准。

### 训练输出

训练结果默认保存到：

```text
logs/<experiment_name>/<date_time>_<run_name>/
├── config.json
├── source/
├── model_0.pt
├── model_100.pt
└── ...
```

其中 `config.json` 保存本次配置，`source/` 保存训练启动时的关键源码快照，便于复现实验。

可使用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir logs
```

---

## 🎮 Play 与恢复评估

### 使用配置文件指定的模型

```bash
python simple_play.py --task pandaN3poHim
```

当未传入模型参数时，程序会读取 `configs/panda_config.py` 中的 `runner.resume_path`。启动后请检查终端输出：

```text
Loading model from: ...
```

确保实际加载的 checkpoint 正确。

### 指定某次运行和 checkpoint

```bash
python simple_play.py \
  --task pandaN3poHim \
  --load_run Aug15_15-55-06_flat_model1000_exact_upside_safe_first_attempt_finetune \
  --checkpoint 8000
```

`--load_run` 是当前 `experiment_name` 对应日志目录下的运行文件夹名称，`--checkpoint` 是 `model_<iteration>.pt` 中的迭代编号。如果运行目录属于其他实验，可同时传入 `--experiment_name <name>`。

### 九宫格测试

Play 默认创建 9 台机器人，并行展示精确朝天、倾斜朝天、左右侧倒、俯仰翻转和不同关节姿态。交互窗口显示全部机器人，录制视频则跟随第 0 台机器人。

键盘控制：

| 按键 | 操作 |
|---|---|
| `0`～`8` | 选择对应编号的机器人 |
| `U` | 将所选机器人设置为四脚朝天，由策略自主恢复 |
| `J` | 将所选机器人设置为左侧倒地 |
| `K` | 将所选机器人设置为右侧倒地 |
| `R` | 设置为随机摔倒姿态和随机关节姿态 |
| `T` | 直接重置到默认站立姿态，不执行策略起身 |
| `F` | 切换九宫格自由相机与机器人跟随相机 |
| `[` / `]` | 选择上一个 / 下一个机器人 |
| `Space` | 暂停 / 继续仿真 |
| `V` | 开启 / 关闭 viewer 同步 |
| `Esc` | 退出 Play |

例如，按 `3` 选择第 3 台机器人，再按 `U`，即可反复测试其从四脚朝天状态自主恢复。

如需恢复单机器人模式，可修改 `simple_play.py`：

```python
RECOVERY_EVAL_NUM_ENVS = 1
USE_RECOVERY_TEST_SUITE = False
```

### 无界面评估

```bash
python simple_play.py --task pandaN3poHim --headless
```

无界面模式会执行有限时长的评估并在终端输出成功率、首次恢复时间以及恢复后的速度和动作变化等指标。

---

## 💾 ONNX 导出

`simple_play.py` 每次成功加载 checkpoint 后都会自动将策略导出到项目根目录：

```text
model.onnx
```

终端会显示：

```text
Exported ONNX policy to: .../model.onnx
```

如果根目录已存在同名文件，该文件会被覆盖。需要保留多个版本时，请在 Play 完成后及时重命名。

---

## 🤖 Sim2Sim 与 Sim2Real

仓库中的 `rl_sar/` 用于 ONNX 策略的仿真和实机部署，Panda 配置位于：

```text
rl_sar/policy/panda/base.yaml
rl_sar/policy/panda/legged_gym/config.yaml
rl_sar/policy/panda/legged_gym/model.onnx
```

部署前需要：

1. 将导出的 `model.onnx` 复制到 `rl_sar/policy/panda/legged_gym/`。
2. 在 `config.yaml` 中确认 `model_name` 与模型文件名一致。
3. 检查观测顺序、10 帧历史观测、动作缩放和控制周期。
4. 检查 `base.yaml` 中的关节名称、`joint_mapping`、PD 参数和力矩限制。
5. 先完成 Gazebo 或 MuJoCo 的 Sim2Sim 验证，再进行实机测试。

`rl_sar` 的完整依赖、编译和启动方式见 [`rl_sar/README_CN.md`](rl_sar/README_CN.md)。以 ROS 2 Gazebo 为例：

```bash
cd rl_sar
./build.sh
source install/setup.bash
ros2 launch rl_sar gazebo.launch.py rname:=panda
```

在新终端启动控制程序：

```bash
cd rl_sar
source install/setup.bash
ros2 run rl_sar rl_sim
```

> ⚠️ **安全提示：** 实机部署前必须逐项核对关节映射、观测顺序、控制频率、PD 增益、动作缩放与力矩限制。首次测试应架空机器人或使用安全吊架，并准备急停。错误配置可能造成机器人突然运动或硬件损坏。

---

## 📂 仓库结构

```text
.
├── train.py                     # 训练入口
├── simple_play.py               # 恢复评估、键盘控制与 ONNX 导出
├── global_config.py             # 项目根路径
├── configs/
│   ├── panda_config.py        # Panda 环境、奖励、恢复与训练配置
│   └── legged_robot_config.py # 通用环境配置
├── envs/                        # Isaac Gym 环境与恢复逻辑
├── algorithm/                   # NP3O 算法
├── modules/                     # Actor-Critic 与历史观测编码网络
├── runner/                      # 训练 runner
├── utils/                       # 参数、日志、地形与任务注册工具
├── resources/
│   └── panda3_2/              # Panda URDF 与 mesh 资源
├── logs/                        # checkpoint、配置和源码快照
├── rl_sar/                      # Gazebo / MuJoCo / 实机部署框架
└── FR-Net/                      # 相关基础项目代码
```

---

## ⚙️ 关键配置

主要配置集中在 `configs/panda_config.py`：

| 配置项 | 当前作用 |
|---|---|
| `env.num_envs` | 并行训练环境数量，默认 4096 |
| `env.episode_length_s` | 单次恢复 episode 时长，默认 12 秒 |
| `init_state` | 初始高度、关节角度和重置速度 |
| `control` | PD 增益、动作缩放和控制降采样 |
| `rewards.scales` | 姿态恢复、稳定、接触与动作约束奖励 |
| `domain_rand` | 摔倒姿态比例、随机化范围和成功判据 |
| `policy` | 网络结构与历史观测编码设置 |
| `runner` | 实验名、运行名、迭代数与续训 checkpoint |

修改奖励或恢复初始分布后，建议先用较少环境和较短迭代进行检查，再启动完整训练。

---

## 📚 参考项目与论文

- [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)
- [Rapid Locomotion via Reinforcement Learning](https://arxiv.org/abs/2205.02824)
- [Extreme Parkour with Legged Robots](https://arxiv.org/abs/2309.14341)
- [Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination](https://arxiv.org/abs/2301.10602)
- [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230)
- [rl_sar](https://github.com/fan-ziqi/rl_sar)

---

## 📋 TODO

- [ ] 补充训练与测试视频
- [ ] 整理可直接下载的推荐 checkpoint
- [ ] 增加各类摔倒姿态的定量成功率对比
- [ ] 完善 MuJoCo Sim2Sim 启动示例
- [ ] 补充实机部署流程与安全检查清单
