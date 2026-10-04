import json
import sys

sys.path.insert(0, '.')

import numpy as np

from sensor_calibration.models import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    Observation,
    ObservationBatch,
    SpatiotemporalScenario,
)
from sensor_calibration.synthetic import make_moving_batch, rotation_about_axis
from sensor_calibration.spatiotemporal import (
    SpatiotemporalConfig,
    calibrate_spatiotemporal,
    compare_spatiotemporal_scenarios,
)

R = rotation_about_axis(np.array([0.3, 0.5, 0.8]), 31.0)
T = np.array([0.4, -0.2, 0.15])
OFFSET = 0.18
times = np.linspace(0.0, 4.0, 61)
conv = CoordinateConvention('c', 'a', 'b', 'meter')
ver = CalibrationVersion('spacetime', '2.0.0')
cfg = SpatiotemporalConfig(search_half_range_seconds=2.0)


def curved(tt):
    return np.array([tt, 0.6 * np.sin(0.9 * tt), 0.4 * tt + 0.3 * np.cos(0.5 * tt)])


# 1. known R/t + 0.18 s offset, non-straight trajectory
b = make_moving_batch(batch_id='curved', trajectory_a=curved, rotation=R, translation=T,
                      sample_times=times, true_offset_seconds=OFFSET, seed=11)
res = calibrate_spatiotemporal(CalibrationInput('d', (b,)), conv, ver, cfg)
assert res.success and res.spatial_status == 'valid', res.reasons
assert res.time_offset.status == 'identifiable'
assert abs(res.time_offset.offset_seconds - OFFSET) < 0.01, res.time_offset.offset_seconds
assert np.linalg.norm(res.transform.rotation - R, ord='fro') < 1e-2
assert np.linalg.norm(res.transform.translation - T) < 1e-2
print('1 curved: offset', round(res.time_offset.offset_seconds, 5),
      'Rerr', round(np.linalg.norm(res.transform.rotation - R, ord='fro'), 6),
      'terr', round(np.linalg.norm(res.transform.translation - T), 6))

# 2a. static target -> unidentifiable, no arbitrary transform
bs = make_moving_batch(batch_id='static', trajectory_a=lambda tt: np.array([1.0, 2.0, 0.5]),
                       rotation=R, translation=T, sample_times=times,
                       true_offset_seconds=OFFSET, seed=12)
rs = calibrate_spatiotemporal(CalibrationInput('d', (bs,)), conv, ver, cfg)
assert rs.time_offset.status == 'unobservable_target_static'
assert rs.time_offset.offset_seconds is None
assert not rs.success and rs.spatial_status == 'undetermined'
print('2a static:', rs.time_offset.status, 'success', rs.success)

# 2b. constant-velocity straight target -> confounded, range not point
bl = make_moving_batch(batch_id='line', trajectory_a=lambda tt: np.array([tt, 0.3 * tt, -0.2 * tt]),
                       rotation=R, translation=T, sample_times=times,
                       true_offset_seconds=OFFSET, seed=13)
rl = calibrate_spatiotemporal(CalibrationInput('d', (bl,)), conv, ver, cfg)
assert rl.time_offset.status == 'confounded_with_motion'
lo, hi = rl.time_offset.feasible_offset_range_seconds
assert hi - lo > 0.5 and lo <= OFFSET <= hi
print('2b straight:', rl.time_offset.status, 'range', (round(lo, 2), round(hi, 2)))

# 3a. missing period -> gap recorded, never bridged, provenance kept
bg = make_moving_batch(batch_id='gapped', trajectory_a=curved, rotation=R, translation=T,
                       sample_times=times, true_offset_seconds=OFFSET, seed=11,
                       b_gap_units=((1.5, 2.5),))
rg = calibrate_spatiotemporal(CalibrationInput('d', (bg,)), conv, ver, cfg)
sel = rg.selection[0]
assert sel.gaps_b_clock, 'gap must be recorded'
gap_lo, gap_hi = sel.gaps_b_clock[0]
for iv in sel.used_intervals_b_clock:
    assert iv.end <= gap_lo or iv.start >= gap_hi, (iv, sel.gaps_b_clock)
assert abs(rg.time_offset.offset_seconds - OFFSET) < 0.01
print('3a gap:', sel.gaps_b_clock, 'used', sel.paired_count,
      'omitted', set(sel.omitted_reasons.values()))

# 3b. window boundary -> excluded observations keep their reason
full, scen = compare_spatiotemporal_scenarios(
    CalibrationInput('d', (b,)), conv, ver,
    {'trim': SpatiotemporalScenario(excluded_a_windows={'curved': ((2.5, 4.0),)})}, cfg)
s = scen[0].result.selection[0]
windowed = [oid for oid, why in s.omitted_reasons.items() if why == 'excluded_time_window']
assert windowed, 'window exclusion must be recorded'
ta = {o.observation_id: o.t_a for o in b.observations}
assert all(2.5 <= ta[oid] <= 4.0 for oid in windowed)
assert all(iv.end <= 2.5 for iv in s.used_intervals_a_clock)
print('3b window:', len(windowed), 'excluded, a-intervals',
      [(iv.start, iv.end) for iv in s.used_intervals_a_clock])

# 3c. clock reset -> segments split, overlap recorded, no arbitrary success
obs = []
rng = np.random.default_rng(5)
for i, t in enumerate(times):
    pa = curved(float(t))
    cb = R @ pa + T
    tb = float(t) - OFFSET
    if t >= 2.0:
        tb -= 1.0  # B clock resets backwards by 1 s at t=2
    obs.append(Observation(
        observation_id=f'obs-{i:03d}', target_id=f't-{i:03d}',
        sensor_a_position=pa, sensor_b_position=cb + rng.normal(0, 1e-4, 3),
        position_sigma=1e-4, t_a=float(t), t_b=tb, sensor_b_position_clean=cb,
    ))
br = ObservationBatch('reset', 'virtual-sensor-a', 'virtual-sensor-b', tuple(obs), 'second')
rr = calibrate_spatiotemporal(CalibrationInput('d', (br,)), conv, ver, cfg)
sre = rr.selection[0]
assert sre.gaps_b_clock, 'reset overlap must be recorded'
assert not rr.success, 'inconsistent reset pairings must not yield a clean success'
assert any('noise' in note for note in rr.identifiability_notes)
print('3c reset: gaps', sre.gaps_b_clock, 'used', sre.paired_count,
      'offset', rr.time_offset.offset_seconds, 'success', rr.success)

# 4. saved result keeps auditable data ranges
saved = json.loads(json.dumps(res.as_dict()))
sel0 = saved['selection'][0]
assert sel0['used_intervals_a_clock'] and sel0['used_intervals_b_clock']
assert saved['time_offset']['feasible_offset_range_seconds']
print('4 saved ranges ok:', sel0['used_intervals_a_clock'][:1],
      saved['time_offset']['feasible_offset_range_seconds'])
print('ALL ACCEPTANCE CHECKS PASSED')
