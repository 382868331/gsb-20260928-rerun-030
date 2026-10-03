import sys; sys.path.insert(0, ".")
import numpy as np
import sensor_calibration.spatiotemporal as sp
from sensor_calibration.models import CalibrationVersion, CoordinateConvention
from sensor_calibration.synthetic import make_moving_batch, rotation_about_axis

R = rotation_about_axis(np.array([0.0, 0.0, 1.0]), 25.0)
T = np.array([0.4, -0.2, 0.1])
times = np.linspace(0.0, 4.0, 41)
ver = CalibrationVersion("x", "2"); conv = CoordinateConvention("c", "a", "b", "meter")

def curved(tt): return np.array([tt, 0.6*np.sin(0.9*tt), 0.4*tt + 0.3*np.cos(0.5*tt)])

sp._DUP_SIZES = {"curve": 1.0}
b = make_moving_batch(batch_id="curve", trajectory_a=curved, rotation=R, translation=T,
                      sample_times=times, true_offset_seconds=0.35, seed=1)
cfg = sp.SpatiotemporalConfig(search_half_range_seconds=2.0)
segs, gaps = sp._b_segments(b, cfg.gap_factor)
sd = [sp._interpolate_segment(s, b.seconds_per_unit) for s in segs]
pairs, omitted, *_ = sp._build_pairs(b, cfg, sd, ())
print("pairs", len(pairs), "omitted", len(omitted))
for dt in [-0.35, -0.2, 0.0, 0.2, 0.35, 0.5]:
    act = sp._active_mask(pairs, dt)
    if int(act.sum()) < 4:
        print("dt", dt, "few", int(act.sum())); continue
    rigid, rms, *_ = sp._evaluate(pairs, dt, act)
    print("dt", dt, "n", int(act.sum()), "rms", round(rms, 6))
p0 = pairs[20]
print("pair20 ta", p0[7], "dtlo/hi", p0[9], p0[10])
