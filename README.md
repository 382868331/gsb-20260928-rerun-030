# 传感器标定实验系统

这个最小 Python 系统从两种虚拟传感器的配对 3D 观察估计外参：

- 输出刚体变换 p_b = R @ p_a + t。
- 使用去质心与 SVD（Kabsch）求相对旋转和平移。
- 观测批次、坐标约定和标定方法版本分别建模，结果保留使用和排除的观察 ID。
- 退化判断按目标几何的数值秩分级（两侧传感器点云都参与）：
  - point（秩 0，集中到一点）或 line（秩 1，共线）时无法确定完整刚体变换，返回 success=False / status="undetermined"，因为绕该直线的旋转不可观测；判定阈值以报告的测量噪声为底噪。
  - plane（秩 2，充分展开的平面）和 volume（秩 3）都可以核对完整 R、t，正常拟合为 status="valid"；平面的第三个奇异值为零是预期现象，条件数只在有效秩内计算。
  - 近共线平面、近平面三维采样、残差相对噪声过大或个别观察主导估计时仍返回变换，但 status="unstable" 并在 warnings 中说明。
  - sampling_recommendations 只指出缺失的空间分布：共线/近共线时补偏离主线或第二平面方向，近平面体积采样时补偏离拟合平面的点；充分平面和充分体积不会被笼统要求“多加点”。
  - 运动尺度相对测量噪声不足时返回 undetermined，不强行给变换。
- 每个可估计结果附带留一法（leave-one-out）影响分析：residual、删除该观察后的旋转/平移变化和归一化影响分；显著大于其余观察且残差超出门限的观察列入 influential_observations。
- compare_observation_subsets 支持对人工排除不同观察的子集分别重算，并与全批次结果比较旋转/平移变化；judgment_basis 记录排除项是否高影响、残差是否下降、告警是否消除及变换变化，所有排除都通过配置传入，原批次不可变。
- 已标记为 quality="bad" 的观察默认排除；也可在重算配置中传入额外排除列表，原批次不修改。

## 多批次联合评估

calibrate_joint 把多次采集的批次放在同一刚体关系下联合求解，并保留逐批对照：

- 权重按“批次等权、批内按 1/sigma² 加权”分配：点数多的一批不会仅凭点数主导联合解，测量更准的批内观察权重更高。
- 逐批单独结果与联合结果并列返回；compatibility 记录每批与联合解的关系：supports_joint_solution、conflicts_with_joint_solution、indeterminate_alone_covered_by_joint 或 insufficient_observations。冲突判定基于各批独立解的中位共识（平移中位数 + 投影平均旋转），而不是与折中联合解直接比较，因此能指出哪一批无法用同一变换解释。
- 某一批只有共线（或更退化）的点时，单批判定为 undetermined，但联合解按合并后的几何秩判断；其他批次补足方向信息时联合仍可 valid，并在 identifiability_notes 中说明。
- 同一采集被重复加入（观察逐位相同）时会被识别为重复采集并共享一份权重，effective_batch_count 只计不同采集，重复数据不能凭空增加证据。
- 联合解与批次顺序无关（内部按批次 ID 排序）；长度单位缩放只按比例缩放平移与噪声阈值，不改变旋转、状态或分布判定。
- compare_joint_scenarios 支持保存的对比方案：每个方案用 JointScenario 声明临时排除的整批或个别对应，原采集不被修改；返回的 JointScenarioComparison 记录相对完整联合解的旋转/平移变化、可辨识状态变化（如 scenario_loses_joint_identifiability）及 judgment_basis 判定依据。
- JointCalibrationResult.as_dict 可序列化保存；observation_weights 与 used/excluded_observations 以 (batch_id, observation_id) 追溯每个点属于哪次采集及其采用决定。

## 运行合成验证

    .venv/bin/python experiments/run_synthetic_validation.py
    .venv/bin/python -m unittest discover -s tests -v

脚本包含以下场景：

1. 正常、有噪声且目标在三维中充分展开的观察。
2. 充分展开的平面目标：返回 valid，且用已知 R、t 核对。
3. 近共线的平面目标：返回 unstable 但仍给出变换，旋转误差明显大于充分展开的情形。
4. 含两个未标记坏观察的批次：返回 unstable 并在 influential_observations 中定位；比较“排除第一个/两个都排除”的子集，后者回到 valid。
5. 明确标记坏观察的批次：默认排除后重算。
6. 严格共线、全部集中到一点、以及运动尺度相对噪声不足的批次：判定 undetermined，不返回外参。
7. 多批次联合：共线批次被其他批次补足方向、偏差批次被共识判定为冲突、重复采集不增加证据，以及“排除偏差批/排除共线批”两个保存方案的比较。

## 可追溯字段

- ObservationBatch：记录批次 ID、两个传感器 ID 和批次内的不可变观察。
- CoordinateConvention：记录两个坐标帧、长度单位和左右手系。
- CalibrationVersion：记录方法名、方法版本、软件版本和关键参数。
- CalibrationResult：记录 status（undetermined/unstable/valid）、distribution（point/line/plane/volume）、使用/排除的 (batch_id, observation_id)、逐观察残差、SVD 奇异值、不可确定原因、不稳定告警以及留一影响分析。
- CalibrationResult：sampling_recommendations 记录定向补点建议；measurement_sigma 保留本次判定使用的测量噪声尺度。
- SubsetComparison：记录人工排除子集的标签、额外排除的观察、子集结果、相对全批次的旋转/平移变化，以及 judgment_basis 判定依据。
- JointCalibrationResult：记录联合解、逐批单独解、批次兼容性、逐观察权重、重复采集分组、被排除批次、可辨识说明和有效批次数。
- JointScenarioComparison：记录保存方案的标签、排除内容、场景联合结果、相对完整联合解的变化及 judgment_basis。

## 刚体关系与采集时间偏移联合辨识

运动目标的对应点带有两台传感器各自的原始时钟读数：`Observation.timestamp_a / timestamp_b` 与批次上的 `time_unit`（s/ms/us/ns，内部统一换算为秒）。对齐过程从不改写原始观测，而是生成引用原始 ID 的 `MatchedObservation`。`sensor_calibration.temporal` 提供 `calibrate_temporal` 与 `compare_temporal_scenarios`，三种模式可直接比较：

- `fixed_offset`：时间偏移由外部给定，只用对齐核对空间关系，不估计偏移。
- `joint_estimate`：在偏移搜索区间内联合求固定时钟偏移和 R、t。约定 `time_b = time_a - offset`；以传感器 A 的采样时刻为锚，在 B 的同一物理时刻做分段线性插值，插值点的方差按两个端点传播后再与 A 的测量方差合成。
- `simultaneous_only`：只使用两台时钟在容差内本就同时的观测，不做任何插值；哪些观测参与核对由返回的 `matched_observations` 与 `excluded_temporal_observations` 完整列出。

可辨识性按物理事实判断，而不是只看残差：

- 目标静止时偏移完全不可观测；匀速直线运动时偏移与运动相位/平移混淆。两类情形都返回 `success=False`、`offset_identifiable=False`，并给出剩余解释范围 `time_offset_equivalence_range_seconds`（残差在 1σ 带内的偏移集合），即使能算出一个使残差很小的偏移也不宣布唯一标定。
- 只有包含加速度/曲率（相对报告的 `position_sigma` 足够显著）的运动才能确定偏移；此时报告偏移的曲率标准差 `time_offset_std_seconds`。搜索剖面在边界最优时显式标记需要扩大搜索区间。
- 短记录若看不到加速度，按不可辨识处理；空间分布本身退化（点/线）时沿用既有的 rank 判断。

时间与证据处理：

- 任一传感器时钟上超过 `gap_factor × 中位采样周期` 的缺口会把轨迹切段（`gaps`）。插值只在同段两个真实采样之间进行，缺口两端不会被当成连续观测跨过，段外不外推；落入缺口或重叠区外的锚点列入排除原因。
- 改变 `time_unit`（秒/毫秒/微秒/纳秒）只改变原始单位下的区间记录，秒为单位的偏移、R、t、状态不变；先按时钟排序，记录顺序变化不改变物理结论。
- 多批次共享同一个固定偏移：批次等权、批内 1/sigma²；逐位相同的重复采集共享一份采集权重，重复加入不增加独立证据。静止/匀速批次可被其他加速批次补足（联合仍可辨识），而各加速批次偏好的偏移彼此超出容差时记录 `batches_disagree_on_fixed_clock_offset`，并给出每批独立偏好 `per_batch_preferred_offset_seconds`，不给出折中唯一答案。
- `TemporalScenario` 保存“排除整批”或“排除某批若干原始时钟时间窗”的方案（窗口端点用该批原始单位）；`compare_temporal_scenarios` 返回偏移、旋转、平移变化及 `scenario_loses_offset_identifiability` 等判定依据，原采集不可变。

追溯字段：

- MatchedObservation：锚点观测 ID、插值两端 B 观测 ID、A/匹配到的 B 时刻、段号、插值系数、插值后的 B 位置及其方差、是否插值。
- AdoptedTimeInterval：每批实际采用的连续时间段（原始时钟单位下的 A/B 起止）和段内参与核对的观测数。
- TemporalCalibrationResult：模式、R、t、偏移及其标准差/等价解释范围、是否被搜索边界截断、空间子结果、采用匹配、采用区间、排除原因、缺口、逐批运动类型（stationary/constant_velocity/accelerated/insufficient）、逐批偏好偏移、告警与说明；`as_dict` 可序列化保存。

运行合成核对：

    .venv/bin/python experiments/run_temporal_validation.py

脚本用已知 R、t、0.12s 偏移和加速曲线核对恢复精度，并包含匀速直线、静止、有意过短记录、含缺失时间段、毫秒时钟、两种冲突运动批次以及“排除整批/排除局部时间窗”的保存方案；结尾断言保存结果能追到匹配与采用区间且原始观测未被改写。
