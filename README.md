# SiteTwin ML 模块

L0-L3分层异常检测管线，当前用模拟数据跑通，接入真实硬件只需替换数据源。

## 目录结构

```
config.yaml          所有阈值配置（含冷启动 collection_days / recheck_interval_hours）
state.py              Snapshot / Alert 数据结构
layers/
  l0_gatekeeper.py     数据守门（离线/低电量/物理范围/卡死检测）
  l1_hard_limits.py    绝对底线（无条件硬阈值）
  l2_context.py        单传感器标签离散化 + 组合场景规则（不含occupancy推断，见下方局限2）
  l3_models.py         LOF 在线推理（百分位标定）
pipeline.py            串联L0-L3，支持消融实验（layers参数裁剪层数）
fusion.py              汇总告警，定级，输出ml_output
converters.py          MQTT/ThingsBoard/串口 -> 统一payload格式的翻译层
thingsboard_client.py  ThingsBoard REST API 客户端（登录/查设备/拉时序，凭证走环境变量）
cold_start.py          冷启动：按站点收集正常数据 → 诊断 → 本地训练L3（见下方"运行"）
ml/
  features.py          特征工程（训练/推理共用同一份逻辑）
  train.py              L3(LOF)离线训练脚本（生产用，超参数固定 n_neighbors=46）
  experiment_utils.py   模型选型实验的共用工具（训练集矩阵加载、内部交叉验证目标函数、画图）
  train_iforest.py      Isolation Forest 单独训练 + Optuna贝叶斯超参数搜索
  train_ocsvm.py         One-Class SVM 单独训练 + Optuna贝叶斯超参数搜索
  train_lof.py            LOF 单独训练 + Optuna贝叶斯超参数搜索
  train_base.py           汇总调用上面三个，产出对比表格（仅训练阶段实验，不含L0-L3推理评估）
simulator/
  generate.py           模拟数据生成器（日夜周期+异常注入）
build_training_data.py  生成两周模拟正常数据，过滤后存盘（手动训练路径可选用）
evaluate.py              消融实验：L1-only / L1+L2 / 全层 对比检出率
main.py                  端到端入口，--source sim|thingsboard，含冷启动训练
```

## 运行

不同站点温度/CO2/电流基线不同，模型不能跨站点复用，所以**没有预置模型**。首次运行走
冷启动：数据 → L0-L2 → 收集本站点正常数据 → 诊断 → 训练L3 → 激活；之后运行直接加载
本地模型。"收集是否够"看数据自身时间跨度（`collection_days` 天），训练前诊断会检查
每个特征是否都变化过（避免拿"设备全程没运行"这种数据训出坏模型）。详见 `cold_start.py`
与 config.yaml 的 `cold_start` 段。

```bash
pip install -r requirements.txt

# 模拟数据：首次运行自动生成7天数据、收集+训练，再用注入异常的一天演示检测；
# 之后再跑就用已训模型。想重新走冷启动，删掉 models/ 再跑即可。
python main.py --source sim

# 真实数据（ThingsBoard）：先设好环境变量和 config.yaml 的 thingsboard 段。
# 首次运行进入收集阶段（缓冲落盘、断点续收），攒够 collection_days 天且诊断通过后
# 训练并转入稳态轮询；诊断没过则每 recheck_interval_hours 小时复查一次。
python main.py --source thingsboard
```

`--source sim` 是默认值，可省略。

### 可选：手动训练 / 评估（不走冷启动）

```bash
python build_training_data.py                                  # 两周模拟正常数据
python -m ml.train --data training_data/normal_snapshots.pkl   # 单独训练LOF到 models/
python evaluate.py                                             # 消融检出率对比
```

## 模型选型实验（可选，不影响上面的生产流程）

`ml/train.py` 是生产用的一次性训练脚本，超参数固定写死在代码里。如果要比较
Isolation Forest / OC-SVM / LOF 三者在当前训练数据上谁更合适，用另一套独立的
实验脚本，不会覆盖 `models/` 下生产模型：

```bash
# 单独跑一个模型的贝叶斯超参数搜索（会弹出这个模型自己的训练分数分布图 + 搜索收敛图）
python3 -m ml.train_iforest --trials 25
python3 -m ml.train_ocsvm --trials 25
python3 -m ml.train_lof --trials 25

# 三个一起跑，只汇总弹一张对比表格
python3 -m ml.train_base --trials 25
```

评判标准是"内部交叉验证"：把训练数据自己切一部分做留出集，看模型对没见过的
正常数据打分是否依然合理（异常比例接近配置的 `contamination`），**不涉及L0-L3
推理、不用注入异常数据评估检出率**——检出率层面的模型对比是另一个独立的、
尚未做的实验，见"已知的待办/局限"第6条。产出会存到 `ml/experiments/`（模型
文件、分数分布图、收敛图、对比表格），跟生产用的 `models/` 目录完全分开。

## 接入真实数据（ThingsBoard）

已实现，用 `--source thingsboard`。要准备的：

- **环境变量**（凭证绝不写进代码/配置）：`TB_USERNAME` / `TB_PASSWORD`
- **config.yaml 的 `thingsboard` 段**：`host` 填真实地址，`device_to_pod` 填设备名→pod 映射，
  `poll_interval_seconds` 轮询间隔
- 数据流：`thingsboard_client.py`（REST API 拉时序）+ `converters.from_thingsboard_timeseries()`
  解析成统一 Snapshot，`pipeline.run(snapshot)` 接口不变

其它数据源（`converters.py` 已支持，接入只改 `main.py` 取数方式）：

- **MQTT**：用 paho-mqtt 订阅，收到消息调用 `converters.from_mqtt_message()`
- **串口应急**：逐行读取，调用 `converters.from_serial_line()`

## 在树莓派上部署

### 依赖

Pi 上实际跑 `main.py` 只需要这几个运行期依赖：

```bash
pip install pyyaml scikit-learn numpy joblib requests
```

`requirements.txt` 里还有 `optuna` / `matplotlib`，只有跑模型选型实验（`ml/train_*`）
才用得到，Pi 生产运行不需要装。

ARM 上装 scikit-learn / numpy 建议用 piwheels 预编译轮子，避免现场编译：Raspberry Pi OS
的系统 `pip` 默认已配 piwheels；自建 venv 可加
`--extra-index-url https://www.piwheels.org/simple`。Python 3.9+。

### 步骤

```bash
# 1. 代码拉到 Pi，进项目目录，装依赖
pip install pyyaml scikit-learn numpy joblib requests

# 2. 配 config.yaml：thingsboard.host、device_to_pod、poll_interval_seconds

# 3. 设凭证环境变量（别写进文件）
export TB_USERNAME=...
export TB_PASSWORD=...

# 4. 运行（首次进冷启动收集，攒够天数+诊断通过后训练并转稳态轮询）
python main.py --source thingsboard
```

### 开机自启（可选，systemd）

只需拉起 `main.py` 一个进程即可，冷启动、训练、轮询都在里面。示例 unit：

```ini
[Unit]
Description=SiteTwin ML
After=network-online.target

[Service]
WorkingDirectory=/home/pi/sitetwin_ml
Environment=TB_USERNAME=xxx
Environment=TB_PASSWORD=xxx
ExecStart=/usr/bin/python3 main.py --source thingsboard
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

冷启动收集缓冲落盘在 `training_data/collection_buffer.pkl`，进程重启能断点续收。

## 已知的待办 / 局限（可直接写进报告）

1. `config.yaml` 里所有阈值都是占位值，需要真实硬件数据到位后重新校准
   （包括本次调试中发现的两个真实踩过的坑：offline_timeout要大于实际采样间隔，
   L3的score_alert_threshold要和contamination量级匹配，否则误报泛滥）。
   `current`（pod_03电流读数）的物理单位目前尚未确认（安培还是毫安），
   `l2_context.py`里`current_label`的分档阈值和`simulator/generate.py`里
   `current`的生成逻辑均为占位，单位确认后需要一并校准，本次暂不处理
2. L2的occupancy（占用）状态机已彻底移除，不再作为任何场景规则的判断依据
   （原因：PIR只能探测动作、无法探测存在；门磁的状态-事件转换在现场不可靠；
   CO2在多设备环境下无法可靠归因于人体呼吸。详见架构说明文档七/八节的完整论证）。
   现有场景规则改为只使用可直接观测的持续状态（如门磁开关状态本身）或跨pod信号互相印证
3. 训练数据目前全部来自模拟器，真实硬件数据到位后需要重新走一遍
   build_training_data.py -> train.py 流程；本次移除occupancy相关的两个L3特征
   （occ_occupied/occ_unoccupied）后，特征维度从13降到11，已重新生成训练数据并重训
4. L3 的严重度不直接来自异常分数，分数只用于排序和对L1/L2告警的置信度加成，
   这是有意的设计（见项目讨论记录中"异常分数≠严重度"的论证）
5. 用户反馈抑制机制（ignore学习）尚未实现，当前是"一次性训练+定期重训"的简化方案
6. L3三个模型（IForest/OC-SVM/LOF）目前只做过"训练阶段的内部交叉验证"比较
   （`ml/train_base.py`，判断谁在没见过的正常数据上泛化得好），尚未做过
   基于L0-L3全流程推理+注入异常数据的检出率对比；后续需要用类似
   evaluate.py的方法，在每个模型独立训练好的最优超参数下重新跑一遍检出率/
   误报率评估，才能得出"哪个模型实际检出能力更强"的结论
