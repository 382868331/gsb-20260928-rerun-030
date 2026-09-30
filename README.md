# 传感器标定实验系统

这个最小 Python 系统从两种虚拟传感器的配对 3D 观察估计外参：

- 输出刚体变换 p_b = R @ p_a + t。
- 使用去质心与 SVD（Kabsch）求相对旋转和平移。
- 观测批次、坐标约定和标定方法版本分别建模，结果保留使用和排除的观察 ID。
- 退化判断基于去质心坐标的几何秩与相对噪声尺度的有效秩，区分三种结果：
  - `valid`：非共线平面（有效秩 2）或三维展开（有效秩 3），变换可核对。
  - `unstable`：仍给出变换，但面外运动低于分辨率、条件数差、残差相对测量噪声过大，
    或留一法（leave-one-out）显示结果对个别观察高度敏感。
  - `undetermined`（success=False）：点数不足、目标集中到一点、共线/近共线，
    或只能由反射拟合；不强行给变换。
- 非共线平面点能唯一约束刚体变换（Kabsch 的反射保护选择保平面法向的旋转），不再被当作运动不足拒绝。
- 结果以稳健残差尺度（MAD）标出疑似错误对应，并给出每个观察的留一法影响分和影响最大的观察列表。
- 已标记为 quality="bad" 的观察默认排除；也可在重算配置中传入额外排除列表，原批次不修改。
- `calibrate_subsets` 对同一批次比较多个人工排除子集（含基线），排除只发生在配置层，原批次保留。

## 运行合成验证

    .venv/bin/python experiments/run_synthetic_validation.py
    .venv/bin/python -m unittest discover -s tests -v

脚本使用已知旋转平移，覆盖：三维展开点组（valid）、充分展开的平面点组（valid，
几何秩 2）、共线点组（undetermined），以及含一个错误对应的点组（unstable，
标出坏点）再人工排除重算（valid），原批次不修改。

## 可追溯字段

- ObservationBatch：记录批次 ID、两个传感器 ID 和批次内的不可变观察。
- CoordinateConvention：记录两个坐标帧、长度单位和左右手系。
- CalibrationVersion：记录方法名、方法版本、软件版本和关键参数。
- CalibrationResult：记录状态、使用/排除的 (batch_id, observation_id)、残差、
  几何秩/有效秩、运动与噪声尺度、SVD 奇异值、疑似坏观察、留一法影响分以及原因。
