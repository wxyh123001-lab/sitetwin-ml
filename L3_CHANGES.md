# L3 层改动记录

本文档记录本轮调试中对 L3（LOF异常检测）相关代码的全部改动，包括触发原因、
根因定位过程和最终方案。按时间顺序排列。

---

## 背景：现象

真实 Pi 部署后，L3 持续以 INFO 级别报 `l3_rare_pattern`，分数长期停留在 0.99
以上，几乎每次轮询都报，明显不正常（正常应该只有约 `contamination`=2% 的比例
被打高分）。用真实 Pi 控制台日志（而不是猜测）逐步定位，一共发现四个独立原因，
前三个是数据/字段层面的问题，第四个是模型/门控层面的问题。

---

## 问题一：轮询窗口内字段稀疏，缺失字段被当成读数0.0（main.py 实时轮询）

**现象**：真实日志显示每次poll的 `readings` 字典只有1~4个字段（总共9个），
其余字段在 `make_features()` 里被 `.get(field, 0.0) or 0.0` 默认成0.0——一个
物理上不可能的值（比如温度0.0℃）随之进入模型，自然被判定为异常。

**根因**：ThingsBoard真实设备不是每次轮询窗口内都会有新读数上报，短轮询窗口
（`poll_interval_seconds`=10秒）经常拿不到某些字段的新数据。

**修复**（`main.py` `_tb_poll_snapshot`）：新增跨poll的 `last_known` 缓存——
每次poll把拿到的新字段更新进缓存，下次poll如果没有新数据，就用缓存里上一次
收到的真实值顶上，不再让字段直接消失。

```python
last_known = dev.setdefault("last_known", {})
for _, fields in parsed:
    last_known.update(fields)
if last_known:
    readings_by_pod[pod_id] = dict(last_known)
```

---

## 问题二：长期被硬件门禁的字段，从未有机会写入 last_known 缓存

**现象**：修完问题一后，pod_03的 `current`（电流）、`vibration_rms`（振动）
依旧经常是0.0。

**根因**：真实硬件配置的告警阈值有问题——`current_ma` 阈值配的是 **-1.0**
（任何正常读数都 ≥ -1.0，永远处于"告警中"状态）；`vibration_rms_g` 阈值配的
是 **0.0**（任何非负读数都触发）。这两个字段在真实设备上**从开机起就一直被
硬件门禁过滤掉**，`last_known` 缓存从来没被真实值填过，缓存本身还是空的，
一样默认成0.0。这是ThingsBoard/硬件侧的阈值配置问题，不是Python代码能修的，
建议找硬件那边确认，或通过TB控制台的"调整阈值"功能改。

**临时缓解**（`main.py` `_seed_last_known` + `_FIELD_SEED_DEFAULTS`）：用训练
数据的字段中位数预置 `last_known` 缓存的初始值，而不是让它保持空/0.0。一旦
该字段真的收到过一次可信读数，缓存会被真实值覆盖，种子值只是"从未见过任何
真实值之前"的占位。

```python
_FIELD_SEED_DEFAULTS = {
    "temperature": 28.4, "humidity": 45.8, "co2": 911.0, "voc_index": 141.6,
    "pir_triggered": False, "light_lux": 16.7, "vibration_rms": 0.006,
    "equip_temp": 27.65, "current": 0.3,
}
```

在 `_tb_setup()`（真实ThingsBoard）和 `_mock_tb_devices()`（模拟ThingsBoard）
里各自调用一次。

---

## 问题三：离线训练数据本身混进了同样的"字段置0"伪影（models分支专属）

在把上面两个实时轮询的修复经验对照检查离线训练数据（`deploy/convert_real_data.py`
生成的 `training_data/real_snapshots.pkl`）时，发现同样的问题也存在于训练数据
本身，而且比实时轮询更严重，因为离线转换脚本当时完全没有last_known/种子填充
的逻辑：

### 3a. 字段级：某字段在某个时刻从未上报过 / 正被门禁

`build_pod_timeline()` 原先的逻辑是：字段还没收到过第一条真实读数，或者当前
正被硬件门禁，就直接把这个字段从 `fields` 字典里删掉（`continue`）。结果同
问题一——被删掉的字段在 `make_features()` 里默认成0.0，实测发现13条快照的
`temperature`是精确的0.0（真实房间温度不可能是0℃）。

**修复**：改成"字段级last_known + 种子默认值"回退，跟实时轮询同一套思路：

```python
if val is not None and not gated:
    last_known[field] = val
    fields[field] = val
elif last_known[field] is not None:
    fields[field] = last_known[field]   # 当前被门禁 -- 用上次可信读数顶上
elif field in _FIELD_SEED_DEFAULTS:
    fields[field] = _FIELD_SEED_DEFAULTS[field]  # 从没收到过可信读数 -- 用中位数占位
```

验证：修复后 `temperature`/`humidity`/`co2`/`voc_index` 精确等于0.0的记录数
全部归零。

### 3b. 整pod级：某个pod还没开始上报任何数据（影响约20%训练数据，比3a严重得多）

真实数据里，pod_02（12:17:13开始）、pod_03（12:19:23开始）比pod_01
（07:58:32开始）晚了**4个多小时**才第一次上报。`main()` 里跨pod合并快照的
逻辑，原先在某个pod完全没开始上报前，把它整体排除在 `readings` 字典外
（`pod_current[pod_id]` 保持空字典 `{}`），导致这4个多小时里所有快照的
pod_02/pod_03字段被 `make_features` 一次性全部置0——`equip_temp==0 and
current==0 and vibration_rms==0` 同时成立的记录，数量远超3a那13条。

**修复**：合并循环开始前，就用种子默认值预置每个pod的字段，而不是空字典：

```python
pod_current = {}
for pod_id in per_pod_timeline:
    pod_fields = {field for (_sid, _cap, field, _rk, _th) in POD_SCHEMA[pod_id]}
    pod_current[pod_id] = {f: _FIELD_SEED_DEFAULTS[f] for f in pod_fields if f in _FIELD_SEED_DEFAULTS}
```

验证：修复后"三字段同时为0"的整pod缺失特征信号，记录数归零。

用修复后的真实数据（9339条，0.85天）单独重新训练了LOF，替换了`models`分支
上部署的模型（此前是真实+模拟混合训练的）。

---

## 问题四：种子/中位数向量本身仍被判定为满分异常（模型/门控层面，非数据bug）

修完问题一~三、确认训练数据里的0.0伪影全部清除后，重放真实日志发现刚启动、
所有字段还停留在种子默认值阶段的poll，依然被打成0.9986这种接近满分的分数。

**根因排查**：种子默认值是**全天24小时的全局中位数**，但真实数据里，很多
字段和时段强相关——比如实测午夜前后 `light_lux` 实际约47.5（不是种子值
16.7），`vibration_rms` 波动很大（中位数0.012、最高到8.4，不是恒定的种子值
0.006），`equip_temp` 约29.4（不是种子值27.65）。种子值是"不分时段的通用值"，
跟当前快照真实的 `hour_sin`/`hour_cos` 组合在一起时，形成了一个训练数据里
从未出现过的**联合状态**——LOF是联合密度方法，看的是"这个多维组合是否常见"，
不会因为每个字段单独看都落在合理范围内就网开一面。

验证：把同一时段一条**真实**训练数据（不是种子值）拿去打分，分位数只有
0.0076（完全正常）——证明模型本身没问题，问题precisely出在"种子默认值不
区分时段"这一点上，会在冷启动/长期门禁的过渡阶段短暂触发误报。

**最终方案**（用户选择）：不做按时段分桶种子值这种更复杂的改法，而是加一个
**范围门控**：只要当前快照的每个字段值，都落在训练数据里"历史上出现过的
最小值~最大值"范围内，就不触发L3告警，不管LOF联合打分多高。种子中位数天然
落在min~max之间，因此会被判定为正常；一个真正超出历史观测范围的值（比如
温度低于任何训练数据见过的读数）依然会正常报警。

`ml/train.py`——训练时额外保存每个特征的[min, max]：

```python
feature_range = np.stack([X.min(axis=0), X.max(axis=0)])
np.save(os.path.join(models_dir, "feature_range.npy"), feature_range)
```

`layers/l3_models.py`——打分时加门控：

```python
def _within_seen_range(self, x_raw):
    if self.feature_range is None:
        return False
    feat_min, feat_max = self.feature_range[0], self.feature_range[1]
    return bool(np.all((x_raw >= feat_min) & (x_raw <= feat_max)))

...
if snapshot.anomaly_score >= self.alert_threshold and not self._within_seen_range(x[0]):
    snapshot.add_alert(...)
```

`anomaly_score` 本身仍然如实计算、保留在 `snapshot.anomaly_scores_by_model`
里供排查/日志使用，只是不满足"范围内"条件才不会触发 `add_alert`。

**验证**：把用户提供的真实日志逐条poll重放：
- 前4条（种子值/接近种子值组合）：不再报警
- 第5条（温度真的降到23.991℃，超出真实数据观测过的26.94~31.0℃范围）：
  依然正常报警——证明门控只压掉"历史范围内但联合罕见"的情况，不会掩盖
  真正超出历史范围的读数

---

## 已知局限

- 范围门控是"每字段独立判断"，不是联合判断——如果未来出现"每个字段单独看
  都在历史范围内，但组合起来确实是真异常"的情况（比如温度和湿度都在正常
  范围内，但组合方式物理上不该同时出现），门控会连这种真异常也一起压掉。
  目前判断是：训练数据只有0.85天，误报的代价（INFO噪音）比漏报更值得优先
  处理，等积累更多真实数据、误报现象消失后可以考虑收紧或去掉这层门控
- 训练数据只有0.85天，观测到的正常范围本身就窄，范围门控在数据量增长后
  会自然变严格（历史min/max范围随训练数据增多而扩大，不需要手动调）
- `main.py`的种子机制解决的是"实时轮询"场景，`convert_real_data.py`的种子
  机制解决的是"离线转换真实历史数据"场景，两边用的是同一份
  `_FIELD_SEED_DEFAULTS`常量，但物理上是两份独立的代码拷贝（避免
  `convert_real_data.py` import `main.py` 带来不必要的耦合/副作用风险），
  以后调整种子值需要两边一起改
