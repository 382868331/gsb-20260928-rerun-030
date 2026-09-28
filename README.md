# 传感器标定实验系统

这个最小 Python 系统从两种虚拟传感器的配对 3D 观察估计外参：

- 输出刚体变换 p_b = R @ p_a + t。
- 使用去质心与 SVD（Kabsch）求相对旋转和平移。
- 观测批次、坐标约定和标定方法版本分别建模，结果保留使用和排除的观察 ID。
- 对点数不足、目标分布退化、运动尺度相对噪声不足的采样返回 success=False，不强行给变换。
- 已标记为 quality="bad" 的观察默认排除；也可在重算配置中传入额外排除列表，原批次不修改。

## 运行合成验证

    .venv/bin/python experiments/run_synthetic_validation.py
    .venv/bin/python -m unittest discover -s tests -v

脚本包含三类场景：

1. 正常、有噪声且目标在三维中充分展开的观察。
2. 含两个明确标记坏观察的批次；先展示包含坏点时的偏差，再排除坏点重算。
3. 目标只沿一条短线移动的批次；系统判定数据不足，不返回外参。

## 可追溯字段

- ObservationBatch：记录批次 ID、两个传感器 ID 和批次内的不可变观察。
- CoordinateConvention：记录两个坐标帧、长度单位和左右手系。
- CalibrationVersion：记录方法名、方法版本、软件版本和关键参数。
- CalibrationResult：记录使用/排除的 (batch_id, observation_id)、残差、SVD 奇异值以及不可确定原因。
