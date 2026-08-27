# Quadruped Robot — Optimization-Based Model Control

基于优化的四足模型控制验证框架（MuJoCo 仿真，pipeline 与实机对齐）。

## 目录

```
assets/urdf/go1_description/   # Go1 URDF + mesh
config/                        # robot / locomotion / mpc
  gait_config/                 # 步态：march.yaml, trot.yaml
quadruped/
  controllers/
    low_level/                 # 步态共用：Joint PD, Balance, WBC, AttitudeAssist
    gait_controller/           # Locomotion (march/walk), MPC
    stand/                       # StandController, stand_up
  planners/                    # 步态、足端、躯干规划
  models/                      # SRBM
  sim/                         # MuJoCo 场景 + 环境
  utils/                       # quintic, Raibert
scripts/
  view_urdf_mujoco.py          # MuJoCo 站立 viewer
  plot_stand_debug.py          # 调试：实时关节角 / 扭矩曲线
  dae_to_stl.py                # DAE → STL 转换
tests/
```

## 快速开始

```bash
cd "Quadruped robot"
pip install -r requirements.txt

pytest tests/ -v
python scripts/view_urdf_mujoco.py

# 站立调试：MuJoCo + 实时关节角 / 扭矩曲线
python scripts/plot_stand_debug.py
```

## 控制 pipeline（Phase 0 站立）

```
断电: τ = 0 → 重力下瘫在地上
通电: q_init → Fuse → Tachi（线性，各 5s）→ hold
      q_des = q_nom(t) + Δq_imu(roll, pitch)   # Phase 1.5
      τ = τ_ff + KpΔq + KdΔdq  （Unitree low-level / SDK 教程）
```

运行 `python scripts/view_urdf_mujoco.py`：

- 启动时 **断电**（τ=0，瘫在地上）
- **500 Hz 控制环**与 **~30 Hz 画面**解耦（`sim_timing`，对齐真机）
- 按 **`9`** 通/断电（`robot.yaml` `power_key`；勿用 **P**，MuJoCo 会切 Contact Split 可视化）
- **Ctrl + 右键拖拽**：对 trunk 施加推力（Simulate 同款 perturb）；双击可选其他 body
- 画面右上角 **IMU** = 姿态外环生效；**GATE** = 倾角过大、站起插值暂停
- 按 **`Home`** 重置 spawn + 断电（勿用 **R**，MuJoCo 会切 Reflection）
- 站稳 (hold) 后按 **`8`** 进入/退出行走；**方向键** 平移，**Insert/Delete** 偏航，**End** 清零速度

MuJoCo viewer 会占用大量单字母键（W=线框、S=阴影、A/D=可视化…），行走键已改到方向键/数字键，见 `config/locomotion.yaml`。

| 阶段 | 内容 |
|------|------|
| 0 | MIT 站立：默认角 + 关节 PD + 重力前馈 |
| 1–3 | 步态 / 足端 / 躯干规划 + **键盘行走 (Step A)** |
| 4 | SRBM-MPC（替换 Balance QP） |
| 5 | WBC（已实现基础版） |

### 原地踏步（当前默认，`mode: march_in_place`）

```
按 8（站稳 hold 后）→ 对角 trot @ 1Hz
落足 = 进入时记录的 stance 锚点（quintic 抬起再落回，无 Raibert 前移）
v_cmd = 0；Balance QP + IK swing
```

后续行走：改 `locomotion.yaml` → `mode: walk` + `gait_config: trot`

### Stand vs Walk（unitree_guide）

```
Stand (hold):  StandController — q_nom + Kp/Kd + τ_g

Walk (500 Hz):
  规划: GaitScheduler → FootPlanner(Raibert 落足 + quintic 摆腿) + TrunkPlanner(v_cmd)
  支撑: BalanceCtrl QP → GRF F → τ = −JᵀF     （关节 Kp=0，软约束 wrench + 摩擦金字塔）
  摆动: foot_ref → LegIK → q_des + 关节 Kp/Kd   （unitree，quintic 保留）
  全腿: τ_g = qfrc_bias + joint Kd
```

### 步态规划（`config/gait_config/`）

| 文件 | 用途 |
|------|------|
| `march.yaml` | 原地踏步 trot，`placement: in_place` |
| `trot.yaml` | 行走 trot，`placement: raibert` |

`locomotion.yaml` 里 `gait_config: march` 或 `trot`（也支持 `gait_config/march.yaml`、旧名 `gait_march.yaml`）。

| 模块 | 作用 |
|------|------|
| `GaitScheduler` | Trot 相位 φ=(f·t+offset) mod 1；φ<swing_ratio 摆动，否则支撑 |
| `FootPlanner` | 支撑：固定落足点；摆动：**stance 足端 + body 系 Raibert 预览** + quintic |
| `TrunkPlanner` | 水平跟踪 v_cmd，高度常值，yaw 积分 |

Trot 对角配对：FL+RR 同相，FR+RL 同相（phase_offset 0 / 0.5）。

## 配置

- `config/robot.yaml` — Go1 参数、`stand_joint` PD、`attitude_assist` IMU 外环、`sim_timing` 控制/显示/打印频率
- `config/gait_config/march.yaml` — 原地踏步步态
- `config/gait_config/trot.yaml` — 行走 trot
- `config/locomotion.yaml` — 键盘速度、Balance QP、WBC 增益
- `config/mpc.yaml` — MPC

## Go1 mesh（首次）

```bash
pip install trimesh pycollada fast-simplification
python scripts/dae_to_stl.py
```
