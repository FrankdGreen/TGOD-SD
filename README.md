# TGOD-SD 模仿学习复现：UR5e + MuJoCo，模块化版本

这个版本把之前集中在一个大文件里的逻辑，按照 SAC 和 TGOD-SD 的职责拆开了。核心思想仍然来自论文《基于引导多样性的护理机器人模仿学习》：

- TGOD 轨迹生成：用隐变量 `z` 表示不同策略/轨迹段，策略写成 `π(a|s,z)`。
- SAC 最大熵探索：SAC 的策略熵项对应 TGOD 目标中的 `H(A|S,Z)`。
- MINE 伪奖励：用 `MINE(Z;S) + MINE(Z;D)` 作为内部伪奖励。
- Sinkhorn Distance 轨迹匹配：训练后采样多个 `z` 生成候选轨迹，选择与专家示范 SD 距离最小的一条。

> 说明：论文实验对象是双臂护理机器人；这里根据你给的 `scene.xml / ur5e.xml` 适配为 UR5e 单臂 6 自由度环境。专家样本 `expert_demo.npy` 默认按 `500×12` 使用：`[x,y,z,rx,ry,rz,vx,vy,vz,wx,wy,wz]`。

## 目录结构

```text
TGOD_SD_UR5e_Modular/
├── data/
│   ├── expert_demo.npy
│   ├── scene.xml
│   └── ur5e.xml
├── outputs/
├── inspect_expert_demo.py
├── train.py
├── requirements.txt
├── README.md
└── tgod_sd/
    ├── configs.py                 # 所有配置 dataclass
    ├── utils.py                   # 随机种子、专家样本读取、旋转转换等工具
    ├── xml_utils.py               # MuJoCo XML 运行时修补
    ├── envs/
    │   └── ur5e_env.py            # UR5e MuJoCo 环境封装
    ├── models/
    │   └── networks.py            # Actor、Double-Q Critic、MINE 网络
    ├── algorithms/
    │   ├── replay_buffer.py       # SAC 经验池
    │   ├── sac.py                 # 标准 SAC 更新逻辑
    │   └── tgod_sac.py            # TGOD 伪奖励 + MINE + SAC
    ├── matching/
    │   └── sinkhorn.py            # Sinkhorn Distance 轨迹匹配
    ├── training/
    │   ├── rollout.py             # 单回合采样
    │   └── trainer.py             # 训练组织器
    └── visualization/
        └── plotting.py            # 轨迹可视化
```

## 安装依赖

```bash
conda create -n tgod-sd python=3.10 -y
conda activate tgod-sd
pip install -r requirements.txt
```

你的 Windows 环境如果已经是 `TGOD`，可以直接：

```bash
pip install mujoco torch numpy matplotlib
```

## 先检查专家样本

```bash
python inspect_expert_demo.py \
  --expert_demo data/expert_demo.npy \
  --output_dir outputs
```

会输出每一维的 `min/max/mean/std`，并保存：

```text
outputs/expert_tcp_xyz.png
```

## 开始训练

论文核心伪奖励版本：

```bash
python train.py \
  --expert_demo data/expert_demo.npy \
  --scene_xml data/scene.xml \
  --ur5e_xml data/ur5e.xml \
  --episodes 2000 \
  --horizon 500 \
  --n_candidates 20 \
  --output_dir outputs
```

如果前期探索太散，可以临时打开一个很小的工程稳定项：

```bash
python train.py \
  --episodes 2000 \
  --anchor_reward_weight 0.03 \
  --output_dir outputs_anchor
```

`anchor_reward_weight` 不是论文核心公式，只是为了让 UR5e 单臂在早期探索时更容易靠近专家 TCP 轨迹附近。

## 输出文件

训练完成后会得到：

```text
outputs/
├── config.json
├── train_log.jsonl
├── tgod_sd_checkpoint.pt
├── last_episode_tcp.npy
├── best_tgod_sd_trajectory.npz
└── expert_vs_tgod_sd_tcp.png
```

其中 `best_tgod_sd_trajectory.npz` 包含：

- `tcp`：最终 SD 匹配出的末端轨迹
- `qpos`：对应 MuJoCo 关节角轨迹
- `action`：SAC 策略输出动作
- `z`：最佳隐变量
- `sinkhorn_distance`：与专家轨迹的 Sinkhorn 距离
- `expert_demo`：专家示范副本

## 每个模块负责什么

### 1. `envs/ur5e_env.py`

负责 MuJoCo 环境。SAC 动作不是直接 TCP 位姿，而是 6 个关节的增量控制：

```text
q_target = q_current + action_scale * action
```

然后把 `q_target` 写入 `data.ctrl`，由 MuJoCo 执行器推进得到下一步 `qpos`。

### 2. `models/networks.py`

包含：

- `TanhGaussianActor`：SAC 策略网络
- `DoubleQCritic`：双 Q 网络
- `MINENet`：互信息估计网络

### 3. `algorithms/sac.py`

只保留标准 SAC 的更新逻辑：

- Critic target backup
- Actor loss
- alpha 自动温度系数
- target critic 软更新

### 4. `algorithms/tgod_sac.py`

把 TGOD 的 MINE 伪奖励接到 SAC 上：

```text
R_t = MINE(z; s_t) + MINE(z; D)
```

更新顺序是：

```text
update MINE → update Critic → update Actor → update alpha → soft update target critic
```

### 5. `matching/sinkhorn.py`

训练结束后，用 Sinkhorn Distance 在多条候选轨迹里选择最佳模仿轨迹。

## 常用调参

- 动作太大：减小 `--action_scale`，例如 `0.02`
- 动作太小：增大 `--action_scale`，例如 `0.08`
- 前期完全乱跑：设置 `--anchor_reward_weight 0.01~0.05`
- MINE 奖励爆炸：减小 `--reward_scale` 或 `--reward_clip`
- 调试速度慢：先用 `--episodes 10 --n_candidates 3` 检查代码通路
- 训练稳定后：再改成 `--episodes 2000 --n_candidates 20`

## 关于 XML 的 assets

你给的 `ur5e.xml` 里引用了 `assets/*.obj`。如果本地没有这些 OBJ 文件，脚本默认会自动生成一个运行时 XML，删除视觉 mesh，保留算法训练所需的关节、执行器、碰撞体和 TCP site。

如果你有完整 assets 文件夹，可以放在 `data/assets/`，然后运行：

```bash
python train.py --patch_mesh_assets never
```
