# 传感器标定实验系统

该原型估计两台虚拟传感器观测同一目标时的相对刚体位姿：

```text
p_b = R * p_a + t
```

其中 `R` 是相对朝向，`t` 是相对位置。求解采用加权 Kabsch/SVD。

## 数据边界

- `ObservationBatch`：不可变观测批次，包含设备 ID、观测 ID、时间、坐标和坏点标记。
- `CoordinateConvention`：独立记录坐标系、单位和左右手约定。
- `CalibrationVersion`：记录本次计算使用的批次哈希、坐标约定哈希、纳入/排除观测 ID、方法和父版本。
- 重算只生成新版本；原始批次不会被修改。
`is_bad=True` 的观测会被明确排除，也可通过 `extra_excluded_ids` 临时追加排除。

## 退化判定

完整三维位置和朝向只有在目标位置覆盖三个线性独立方向时才可唯一确定。少于三组点、共线、共面或移动幅度过小都会返回：

- `determined=False`
- `transform=None`
- 有效秩、奇异值和不可定原因

`noise_scale` 用于把噪声量级纳入有效秩判断；共面但带微小噪声的采样不会因为数值噪声被误判为三维充分采样。

## 运行

```bash
python -m pytest
python scripts/demo_calibration.py
```

演示使用已知旋转和平移生成有噪声数据，覆盖正常样本、明确标记离群值后重算、目标移动不足三类场景。
