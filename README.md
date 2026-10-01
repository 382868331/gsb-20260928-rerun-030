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

## 多批次联合标定

多次采集通过 `CalibrationInput(batches=...)` 一起评估，所有批次共享同一个刚体关系，但精度和偏差可以不同：

- 联合解使用逆方差加权 Kabsch：按每个观察报告的 `position_sigma` 转换到约定单位后赋权，精度差的批次不会因为点数多就主导结果。
- 每个批次单独给出 `BatchEvaluation`：独立拟合状态、独立几何分布、联合残差均值/RMS、无法用共同变换解释的残差偏置 `joint_bias_norm` 及其 `bias_sigma_ratio`、与联合解的旋转/平移差异，以及 `supports_joint_transform`。批次偏置或独立解与联合解显著不一致时在 warnings 中逐批标出。
- 可辨识性在合并后的点云上判断：某一批只有共线点（单独 undetermined）但另一批提供了缺失方向时，联合解仍可成功，该批次的 `reasons` 明确说明方向信息由其他批次补足；单批不可靠不等于联合失败。
- 单位尺度：`ObservationBatch.length_scale_to_convention` 与 `length_unit` 在拟合前把该批所有位置和 sigma 换算到约定单位（如毫米批用 0.001），不同单位的批次可直接合并。
- 批次顺序不影响联合解（加权 SVD 与批次顺序无关）。
- 重复采集：批次可用 `acquisition_id` 标识同一次物理采集；相同 `(acquisition_id, target_id)` 的多个副本只算作一份独立证据（证据权重在副本间平分），重复加入不会凭空增加证据，并以 `repeated_acquisition_copies_are_counted_once` 警告提示。
- 暂时排除：`CalibrationConfig(excluded_batches=...)` 可整体排除批次，`additional_excluded_observations` 可排除个别对应；原采集保持不变，结果的 `observation_decisions` 对每个点记录 used / excluded_marked_bad / excluded_manually / excluded_batch 决定和证据权重。整批排除后联合解是否仍可辨识（例如回到 line → undetermined）在合并结果中直接体现。
- 方案保存：`sensor_calibration.persistence` 提供 `build_plan` 与 `save_comparison_bundle`，把排除方案、命名子集比较、逐批原始观察、联合结果、逐点采用决定和采集归属写入 JSON（`schema: sensor-calibration-comparison-bundle/1.0`），之后可追溯每个点属于哪次采集及其采用决定。

新增字段：

- CalibrationResult：`effective_measurement_sigma`、`batch_evaluations`、`observation_decisions`、`acquisition_evidence`。
- ObservationBatch：`length_scale_to_convention`、`length_unit`、`acquisition_id`。
- `BatchEvaluation`、`ObservationDecision`、`AcquisitionEvidence`、`ComparisonPlan`。

`tests/test_multi_batch.py` 覆盖：共线批被其他批补足方向、点多但有偏批次不主导、批次顺序不变性、毫米/米混批、整批排除后可辨识性变化、重复采集只算一份、保存方案的逐点溯源。

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

## 可追溯字段

- ObservationBatch：记录批次 ID、两个传感器 ID 和批次内的不可变观察。
- CoordinateConvention：记录两个坐标帧、长度单位和左右手系。
- CalibrationVersion：记录方法名、方法版本、软件版本和关键参数。
- CalibrationResult：记录 status（undetermined/unstable/valid）、distribution（point/line/plane/volume）、使用/排除的 (batch_id, observation_id)、逐观察残差、SVD 奇异值、不可确定原因、不稳定告警以及留一影响分析。
- CalibrationResult：sampling_recommendations 记录定向补点建议；measurement_sigma 保留本次判定使用的测量噪声尺度。
- SubsetComparison：记录人工排除子集的标签、额外排除的观察、子集结果、相对全批次的旋转/平移变化，以及 judgment_basis 判定依据。
