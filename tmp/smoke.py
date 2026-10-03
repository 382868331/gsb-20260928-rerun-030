import sys
sys.path.insert(0, ".")
import json
import random
import numpy as np
from sensor_calibration.models import (
    CalibrationInput, CalibrationVersion, CoordinateConvention,
    ObservationBatch, SpatiotemporalScenario,
)
from sensor_calibration.synthetic import make_moving_batch, rotation_about_axis
from sensor_calibration.spatiotemporal import (
    SpatiotemporalConfig, calibrate_spatiotemporal,
    compare_spatiotemporal_approaches, compare_spatiotemporal_scenarios,
)

R = rotation_about_axis(np.array([0.0, 0.0, 1.0]), 25.0)
T = np.array([0.4, -0.2, 0.1])
OFFSET = 0.35
times = np.linspace(0.0, 4.0, 41)
conv = CoordinateConvention("c", "a", "b", "meter")
ver = CalibrationVersion("spacetime", "2.0.0")


def curved(tt):
    return np.array([tt, 0.6 * np.sin(0.9 * tt), 0.4 * tt + 0.3 * np.cos(0.5 * tt)])


def straight(tt):
    return np.array([1.0 * tt, 0.3 * tt, -0.2 * tt])


def straight2(tt):
    return np.array([0.2 * tt, 1.0 * tt, 0.1 * tt])


def static_traj(tt):
    return np.array([1.0, 2.0, 0.5])


cfg = SpatiotemporalConfig(search_half_range_seconds=2.0)

b1 = make_moving_batch(batch_id="curved", trajectory_a=curved, rotation=R, translation=T,
                       sample_times=times, true_offset_seconds=OFFSET, seed=1)
res = calibrate_spatiotemporal(CalibrationInput("d", (b1,)), conv, ver, cfg)
print("curved", res.success, res.spatial_status, res.time_offset.status,
      res.time_offset.offset_seconds, res.time_offset.sigma_seconds, res.rms_residual,
      len(res.used_observations))
print("Rerr", np.linalg.norm(res.transform.rotation - R, ord="fro"),
      "terr", np.linalg.norm(res.transform.translation - T))

bs = make_moving_batch(batch_id="straight", trajectory_a=straight, rotation=R, translation=T,
                       sample_times=times, true_offset_seconds=OFFSET, seed=2)
rs = calibrate_spatiotemporal(CalibrationInput("d", (bs,)), conv, ver, cfg)
print("straight", rs.time_offset.status, rs.time_offset.offset_seconds,
      rs.time_offset.feasible_offset_range_seconds, rs.rms_residual)

bt = make_moving_batch(batch_id="static", trajectory_a=static_traj, rotation=R, translation=T,
                       sample_times=times, true_offset_seconds=OFFSET, seed=3)
rt = calibrate_spatiotemporal(CalibrationInput("d", (bt,)), conv, ver, cfg)
print("static", rt.time_offset.status, rt.spatial_status, rt.rms_residual)

bg = make_moving_batch(batch_id="gapped", trajectory_a=curved, rotation=R, translation=T,
                       sample_times=times, true_offset_seconds=OFFSET, seed=1,
                       b_gap_units=((1.5, 2.5),))
rg = calibrate_spatiotemporal(CalibrationInput("d", (bg,)), conv, ver, cfg)
sel = rg.selection[0]
print("gapped used", sel.paired_count, "gaps", sel.gaps_b_clock,
      "omitted", len(sel.omitted_a_observations), set(sel.omitted_reasons.values()))

bms = make_moving_batch(batch_id="curved_ms", trajectory_a=curved, rotation=R, translation=T,
                        sample_times=times, true_offset_seconds=OFFSET, seed=1,
                        time_unit="millisecond")
rms2 = calibrate_spatiotemporal(CalibrationInput("d", (bms,)), conv, ver, cfg)
print("ms", rms2.time_offset.offset_seconds,
      np.linalg.norm(rms2.transform.rotation - R, ord="f"))

obs = list(b1.observations)
random.Random(7).shuffle(obs)
bsh = ObservationBatch("curved", b1.sensor_a_id, b1.sensor_b_id, tuple(obs), b1.time_unit)
ro = calibrate_spatiotemporal(CalibrationInput("d", (bsh,)), conv, ver, cfg)
print("order diff", abs(ro.time_offset.offset_seconds - res.time_offset.offset_seconds))

bdup = ObservationBatch("curved2", b1.sensor_a_id, b1.sensor_b_id, b1.observations, b1.time_unit)
rd = calibrate_spatiotemporal(CalibrationInput("d", (b1, bdup)), conv, ver, cfg)
print("dup", rd.effective_batch_count, rd.duplicate_groups)

bs2 = make_moving_batch(batch_id="straight2", trajectory_a=straight2, rotation=R, translation=T,
                        sample_times=times, true_offset_seconds=OFFSET, seed=5)
rj = calibrate_spatiotemporal(CalibrationInput("d", (bs, bs2)), conv, ver, cfg)
print("two-straight", rj.time_offset.status, rj.time_offset.offset_seconds,
      rj.time_offset.sigma_seconds)

joint, comps = compare_spatiotemporal_approaches(
    CalibrationInput("d", (b1,)), conv, ver, fixed_offset_seconds=0.0, config=cfg)
for c in comps:
    print("comp", c.label, c.result.time_offset.status, c.offset_change_seconds,
          len(c.result.used_observations))

full, scen = compare_spatiotemporal_scenarios(
    CalibrationInput("d", (b1,)), conv, ver,
    {"trim": SpatiotemporalScenario(excluded_a_windows={"curved": ((2.5, 4.0),)})}, cfg)
print("scenario", scen[0].result.time_offset.status, scen[0].judgment_basis[:3])
json.dumps(res.as_dict())
print("json ok")
