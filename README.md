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
