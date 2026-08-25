# 模拟数据阈值汇总

本文档汇总当前代码中"模拟数据"相关的全部阈值，分为四部分：
1. `simulator/generate.py` —— 数据生成基线（决定模拟出来的"正常值"长什么样）
2. `simulator/tb_mock.py` POD_SCHEMA —— 硬件告警网关阈值（超过就整条读数被"门控"丢弃，不进入 Snapshot）
3. `config.yaml` —— L2 场景规则 / L3 阈值（消费数据后做判断的阈值）
4. `layers/l2_context.py` —— 硬编码阈值（不在 config.yaml 里）

生成日期：2026-08-26

---

## 1. `simulator/generate.py` —— 数据生成基线

| Pod | 字段 | 基线 / 公式 | 说明 |
|---|---|---|---|
| pod_01 | temperature | `29.0 + 2.0·sin((hour-6)/24·2π)` | 日夜波动，均值 29，振幅 ±2 |
| pod_01 | humidity | `45 + N(0,3)` | 与温度、时间无关联生成 |
| pod_01 | co2 | 有人 1014.0 / 无人 480.0（目标值，含惯性趋近） | |
| pod_01 | voc_index | `140 + (30 if 有人 else 0) + N(0,5)` | |
| pod_02 | light_lux | `300（有人 or 工作时段） / 17.5（否） + N(0,10)` | 工作时段 = 8:00-19:00，不再区分周末 |
| pod_02 | pir_triggered | 布尔，由"有人"真值驱动 | |
| pod_02 | door_state | 布尔，场景事件驱动（door_left_open 等） | |
| pod_03 | current | 运行时 `120.0+N(0,20)`，空闲 `max(0,N(0,2))` | |
| pod_03 | vibration_rms | 运行时 `0.3+N(0,0.02)`，空闲 `max(0,N(0,0.005))` | |
| pod_03 | equip_temp | `_temp_curve(hour) + (5 if 运行 else 0) + N(0,0.5)` | 复用 pod_01 同一温度曲线，基线也是 29 |
| — | is_work_hour | `8 ≤ hour ≤ 19` | 已去掉"周一到周五"限制，周末与工作日同规则 |

实测（3天样本）峰值：temperature max 32.65，voc_index max 187.3，equip_temp max 37.21。

---

## 2. `simulator/tb_mock.py` POD_SCHEMA —— 硬件告警网关阈值

超过阈值时，该字段本次读数被判定为"当前处于告警状态"，不会写入 Snapshot（模拟真实硬件网关行为）。

| Pod | 能力 (sensor_id) | 字段 | 阈值 | 备注 |
|---|---|---|---|---|
| pod_01 | sht41_temperature | temperature | 高于 **35.0** ℃ | 曾是 30.0，提高避免误网关 |
| pod_01 | sht41_humidity | humidity | 高于 **60.0** % | 未改动 |
| pod_01 | scd41_co2 | co2 | 高于 **1000.0** ppm | 未改动 |
| pod_01 | sgp40_voc | voc_index | 高于 **200.0** | 曾是 80.0（几乎100%被网关掉） |
| pod_02 | bh1750_illuminance | light_lux | 高于 **800.0** lux | 未改动 |
| pod_02 | pir_motion | pir_triggered | 无阈值（布尔直传） | |
| pod_02 | reed_contact | door_state | 触点值 ≥ **1.5** 视为激活 | 曾是 1.0（每次开门都误网关） |
| pod_03 | ds18b20_temperature | equip_temp | 高于 **40.0** ℃ | 曾是 30.0（约42%被网关掉） |
| pod_03 | ina219_current | current | 高于 **400.0** mA | 未改动 |
| pod_03 | adxl345_vibration | vibration_rms | 高于 **1.0** g | 未改动 |

均已重新测量确认：当前基线下上述阈值在正常模拟数据中网关触发率为 0%。

---

## 3. `config.yaml` —— L2 场景规则 / L3 阈值

| 分类 | 键 | 值 |
|---|---|---|
| hard_limits（孤儿配置，仅曾被已删除的 L1 使用，未清理） | temperature_critical | 32 ℃ |
| | co2_critical | 2500 ppm |
| | equip_temp_critical | 70 ℃ |
| | vib_abnormal_rms | 1.0 g |
| labels | temperature | low 18 / high 26 |
| | humidity | low 30 / high 60 |
| | co2 | normal 800 / elevated 1500 / high 2500 |
| | voc | elevated 100 / high 250 |
| | light_lux_on | 50 |
| scenarios.equip_high_load | severity | info |
| scenarios.equip_stall_risk | severity | critical |
| scenarios.vibration_mechanical_fault | severity | warning（现在只看 vib_label=="abnormal"，不再看电流） |
| scenarios.door_left_open | duration_minutes | 20 |
| scenarios.equip_overheat_air_quality | window_minutes / equip_temp_rise_c / current_rise_ratio | 15 分钟 / 5℃ / 0.2 |
| scenarios.equip_cooling_failure | window_minutes / equip_temp_rise_c / current_max_change_ratio | 15 分钟 / 5℃ / 0.05 |
| scenarios.rapid_multi_signal_spike | window_minutes / temperature_jump_c / equip_temp_jump_c / vibration_jump | 5 分钟 / 3℃ / 5℃ / 0.3 |
| l3 | contamination | 0.02 |
| l3 | score_alert_threshold | 0.98 |

---

## 4. `layers/l2_context.py` —— 硬编码阈值（不在 config.yaml 里）

| 变量 | 区间 |
|---|---|
| current_label | off < 10 mA，normal < 400 mA，high ≥ 400 mA |
| vib_label | none < 0.02，normal < 1.0（取自 hard_limits.vib_abnormal_rms），abnormal ≥ 1.0 |
