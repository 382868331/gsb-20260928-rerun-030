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
