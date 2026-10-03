import sys; sys.path.insert(0, ".")
import numpy as np
import sensor_calibration.spatiotemporal as sp
from sensor_calibration.synthetic import make_moving_batch, rotation_about_axis
from sensor_calibration.models import CalibrationVersion, CoordinateConvention
R = rotation_about_axis(np.array([0.,0.,1.]),25.); T=np.array([.4,-.2,.1])
times=np.linspace(0,4,41)
def curved(tt): return np.array([tt,0.6*np.sin(.9*tt),.4*tt+.3*np.cos(.5*tt)])
sp._DUP_SIZES = {"curve":1.0}
b = make_moving_batch(batch_id="curve", trajectory_a=curved, rotation=R, translation=T,
                      sample_times=times, true_offset_seconds=.35, seed=1)
cfg = sp.SpatiotemporalConfig(search_half_range_seconds=2.0)
segs,gaps = sp._b_segments(b, cfg.gap_factor)
sd = [sp._interpolate_segment(s, b.seconds_per_unit) for s in segs]
pairs,om,*_ = sp._build_pairs(b,cfg,sd,())
act = sp._active_mask(pairs,-0.5)
print("active", act.sum())
w = sp._equal_share_weights(pairs, sp._DUP_SIZES, act)
print("sumw", w.sum(), "nan", np.isnan(w).any(), "active sum", w[act].sum())
print("sigmas", [pairs[i][5] for i in np.flatnonzero(act)][:5])
