# SiteTwin ML 模块

L0-L3分层异常检测管线。L0（数据质量把关）/L1（硬阈值告警）由硬件侧负责，本仓库的
生产流程只跑 L2（场景规则）+ L3（LOF异常检测），见下方"运行"。真实 ThingsBoard 数据的
字段映射/解析/硬件报警门禁逻辑（`converters.py`）已经拿真实抓包（`pod1/2/3.json`）验证过，
但还没连过真实 ThingsBoard 实例本身（HTTP 请求那层，`thingsboard_client.py`）。

## 目录结构

```
config.yaml          所有阈值配置（含冷启动 collection_days / recheck_interval_hours）
state.py              Snapshot / Alert 数据结构（Alert 支持 pod_id 单pod 或 pod_ids 跨pod）
layers/
  l0_gatekeeper.py     数据守门（离线/低电量/物理范围/卡死检测）——生产流程不调用，硬件侧已处理，代码留着备用
  l1_hard_limits.py    绝对底线（无条件硬阈值）——同上，生产流程不调用
  l2_context.py        单传感器标签离散化 + 组合场景规则（不含occupancy推断，见下方局限2）
  l3_models.py         LOF 在线推理（百分位标定）
pipeline.py            串联各层，支持消融实验（layers参数裁剪层数，生产用 ["L2","L3"]）
fusion.py              汇总告警，定级，输出ml_output
converters.py          MQTT/ThingsBoard/串口 -> 统一payload格式的翻译层。ThingsBoard部分已用真实
                       抓包验证：字段用sensor_id（非opaque slot）、equip_temp按pod消歧义、硬件
                       alarm _active=true时门禁过滤（不读_observed）、布尔字段按真实"1.0"/"0.0"编码
thingsboard_client.py  ThingsBoard REST API 客户端（登录/查设备/拉时序/建告警，凭证走环境变量）
cold_start.py          冷启动：按站点收集正常数据 → 诊断 → 本地训练L3（见下方"运行"）
ml/
  features.py          特征工程（训练/推理共用同一份逻辑）
  train.py              L3(LOF)离线训练脚本（生产用，超参数固定 n_neighbors=46）
  experiment_utils.py   模型选型实验的共用工具（训练集矩阵加载、内部交叉验证目标函数、画图）
  train_iforest.py      Isolation Forest 单独训练 + Optuna贝叶斯超参数搜索
  train_ocsvm.py         One-Class SVM 单独训练 + Optuna贝叶斯超参数搜索
  train_lof.py            LOF 单独训练 + Optuna贝叶斯超参数搜索
  train_base.py           汇总调用上面三个，产出对比表格（仅训练阶段实验，不含全流程推理评估）
simulator/
  generate.py           模拟数据生成器（日夜周期+异常注入），直接产出Snapshot
  tb_mock.py             假ThingsBoard客户端，把同一份模拟数据包成真实JSON报文格式，走
                        converters.py真实解析/门禁逻辑（--source sim-thingsboard 用）
build_training_data.py  生成两周模拟正常数据，过滤后存盘（手动训练路径可选用）
evaluate.py              消融实验：对比不同层组合的检出率（历史遗留配置仍含L0/L1，跟当前
                        生产的L2+L3配置不完全对应，见下方局限7）
main.py                  端到端入口，--source sim|sim-thingsboard|thingsboard，含冷启动训练
deploy/
  install_service.sh    一键注册成 Pi 上的 systemd 服务（开机自启+崩溃重启），见下方"开机自启"
```

## 运行

不同站点温度/CO2/电流基线不同，模型不能跨站点复用，所以**没有预置模型**。首次运行走
冷启动：数据 → L2（L0/L1 不跑，硬件那边已经处理数据质量把关和硬阈值告警）→ 收集本站点
正常数据 → 诊断 → 训练L3 → 激活；之后运行直接加载本地模型。"收集是否够"看数据自身时间
跨度（`collection_days` 天），训练前诊断会检查每个特征是否都变化过（避免拿"设备全程没
运行"这种数据训出坏模型）。详见 `cold_start.py` 与 config.yaml 的 `cold_start` 段。

```bash
pip install -r requirements.txt

# 模拟数据：首次运行自动生成7天数据、收集+训练，再用注入异常的一天演示检测；
# 之后再跑就用已训模型。想重新走冷启动，删掉 models/ 再跑即可。
python main.py --source sim

# 同样是模拟数据，但走真实 ThingsBoard JSON 报文格式 + converters.py 的真实解析/
# 硬件报警门禁逻辑（不是直接造 Snapshot），用来验证解析代码本身，不用连真实实例。
# 只跑正常数据（异常注入测试是 evaluate.py 的事，这里不做）。
python main.py --source sim-thingsboard

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
尚未做的实验，见"已知的待办/局限"第7条。产出会存到 `ml/experiments/`（模型
文件、分数分布图、收敛图、对比表格），跟生产用的 `models/` 目录完全分开。

## 接入真实数据（ThingsBoard）

已实现，用 `--source thingsboard`。要准备的：

- **环境变量**（凭证绝不写进代码/配置）：`TB_USERNAME` / `TB_PASSWORD`
- **config.yaml 的 `thingsboard` 段**：`host` 填真实地址，`device_to_pod` 填设备名→pod 映射，
  `poll_interval_seconds` 轮询间隔，`severity_map` 是我们severity→ThingsBoard告警severity的
  占位映射（还没最终确认，见代码注释）
- 数据流：`thingsboard_client.py`（REST API 拉时序/建告警）+
  `converters.build_sensor_field_map()`/`from_thingsboard_timeseries()` 解析成统一 Snapshot，
  `pipeline.run(snapshot)` 接口不变。真实读数只用 `{sensor_id}` 原始字段，硬件
  `alarm_{capability}_{rule_kind}_active` 为 true 时那个字段的读数直接过滤掉（不进L2/L3），
  `_observed`等alarm组其它字段一律不请求
- **验证状态**：字段映射、消歧义、门禁过滤这套解析逻辑已经拿真实抓包（pod1/2/3.json）
  测过，但 `thingsboard_client.py` 的 HTTP 请求本身还没连过真实 ThingsBoard 实例，
  想先验证解析逻辑本身可以用 `--source sim-thingsboard`（不用连真实实例）

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

### 开机自启（systemd）

只需拉起 `main.py` 一个进程即可，冷启动、训练、轮询都在里面。用 `deploy/install_service.sh`
一键注册成 systemd 服务（建独立 venv 装依赖、生成凭证文件、写 unit、开机自启+崩溃自动重启），
不用手动抄 unit 文件：

```bash
sudo ./deploy/install_service.sh
```

脚本做的事：
- 在项目目录下建 `.venv`，只装运行期依赖（不含 optuna/matplotlib）——独立 venv 是因为新版
  Raspberry Pi OS（Bookworm 起）默认禁止 `pip install` 到系统环境（PEP 668）
- 凭证写到 `/etc/sitetwin-ml.env`（权限 600，只有 root 能读），systemd 用 `EnvironmentFile=`
  加载，不进 unit 文件（unit 文件通常全局可读）——**第一次跑完脚本会提示你去填
  TB_USERNAME/TB_PASSWORD**，填完执行 `sudo systemctl start sitetwin-ml`
- 写 `/etc/systemd/system/sitetwin-ml.service`：`After=network-online.target` 开机联网后启动，
  `Restart=on-failure` 崩溃自动重启
- 幂等，可重复执行；不会覆盖已存在的凭证文件

常用命令：
```bash
systemctl status sitetwin-ml     # 看服务状态
journalctl -u sitetwin-ml -f     # 看实时日志（告警输出都在这里）
sudo systemctl restart sitetwin-ml
```

冷启动收集缓冲落盘在 `training_data/collection_buffer.pkl`，进程重启能断点续收。

## 已知的待办 / 局限（可直接写进报告）

1. `config.yaml` 里的阈值多数还是占位值，需要真实硬件数据到位后重新校准
   （包括本次调试中发现的两个真实踩过的坑：offline_timeout要大于实际采样间隔，
   L3的score_alert_threshold要和contamination量级匹配，否则误报泛滥）。
   单位已确认：`current`是mA，`vibration_rms`是**g**（按真实抓包的capability名
   `vibration_rms_g`确认，此前一度按m/s²理解是错的，已在代码里改成g，但
   `hard_limits.vib_abnormal_rms`和`vib_label`分档的具体数值还是占位，
   需要拿ADXL345真实基线重新定）
2. 真实设备当前配置的硬件报警阈值，有几个明显没校准/像占位值（比如电流阈值
   -1.0mA、振动阈值0.0g——任何正常读数都会触发），拿这些真实阈值套到模拟的
   正常数据上测试发现会100%触发门禁过滤，导致对应字段收集不到有效数据。
   `simulator/tb_mock.py`里这几个阈值换成了跟我们自己系统一致的边界（不是
   照抄真实值），但如果真实站点的硬件阈值就是这样，真实冷启动大概率会卡在
   同样的地方——上线前建议先跟硬件那边确认这几个阈值改没改
3. L2的occupancy（占用）状态机已彻底移除，不再作为任何场景规则的判断依据
   （原因：PIR只能探测动作、无法探测存在；门磁的状态-事件转换在现场不可靠；
   CO2在多设备环境下无法可靠归因于人体呼吸。详见架构说明文档七/八节的完整论证）。
   现有场景规则改为只使用可直接观测的持续状态（如门磁开关状态本身）或跨pod信号互相印证
4. 训练数据目前全部来自模拟器，真实硬件数据到位后需要重新走一遍
   build_training_data.py -> train.py 流程；本次移除occupancy相关的两个L3特征
   （occ_occupied/occ_unoccupied）后，特征维度从13降到11，已重新生成训练数据并重训
5. L3 的严重度不直接来自异常分数，分数只用于排序和对L2告警的置信度加成，
   这是有意的设计（见项目讨论记录中"异常分数≠严重度"的论证）
6. 用户反馈抑制机制（ignore学习）尚未实现，当前是"一次性训练+定期重训"的简化方案
7. L3三个模型（IForest/OC-SVM/LOF）目前只做过"训练阶段的内部交叉验证"比较
   （`ml/train_base.py`，判断谁在没见过的正常数据上泛化得好），尚未做过
   基于全流程推理+注入异常数据的检出率对比；后续需要用类似
   evaluate.py的方法，在每个模型独立训练好的最优超参数下重新跑一遍检出率/
   误报率评估，才能得出"哪个模型实际检出能力更强"的结论
8. `evaluate.py` 的消融对比配置（`["L0","L1"]`/`["L0","L1","L2"]`/`["L0","L1","L2","L3"]`）
   是L0/L1移出生产流程之前定的，还没跟着改，现在不代表真实部署配置（真实是只有
   `["L2"]`/`["L2","L3"]`两档）；这份对比要不要重新对齐生产配置，还没定
9. `ina219_voltage`（pod_03，真实抓包里确认存在、配了硬件报警规则）目前完全没接：
   `converters.py`没有对应的内部字段，直接跳过。要不要加`voltage`字段，涉及
   `features.py`/`state.py`/L2场景改动，本次先不做（用户确认过跳过）
