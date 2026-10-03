from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sensor_calibration import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    TemporalCalibrationConfig,
    TemporalScenario,
    calibrate_temporal,
    compare_temporal_scenarios,
)
from sensor_calibration.synthetic import make_moving_target_batch, rotation_about_axis


CONVENTION = CoordinateConvention("temporal-convention", "frame-a", "frame-b", "m")
VERSION = CalibrationVersion("temporal-kabsch", "1.0.0")
ROTATION = rotation_about_axis((0.0, 0.0, 1.0), 35.0)
TRANSLATION = np.array([0.5, -0.25, 0.1])
TRUE_OFFSET = 0.12
NOISE = 1e-4


def curved(time_value):
    return np.array([
        0.9 * time_value,
        0.55 * np.sin(1.7 * time_value),
        0.35 * np.cos(1.3 * time_value),
    ])


def straight(time_value):
    return np.array([0.7 * time_value, -0.2 * time_value, 0.3 + 0.1 * time_value])


def stationary(_time_value):
    return np.array([0.1, -0.2, 0.3])


def make_batch(label, trajectory, times, seed, offset=TRUE_OFFSET, unit="s", missing=()):
    return make_moving_target_batch(
        rotation=ROTATION,
        translation=TRANSLATION,
        trajectory_a=trajectory,
        sample_times=times,
        clock_offset=offset,
        noise_sigma=NOISE,
        seed=seed,
        batch_id=label,
        time_unit=unit,
        missing_time_windows=missing,
    )


def report(title, result):
    print(f"\n== {title}")
    print(f"mode={result.mode} success={result.success} offset_identifiable={result.offset_identifiable}")
    print(f"offset_s={result.time_offset_seconds} std_s={result.time_offset_std_seconds}")
    print(f"equivalence_range_s={result.time_offset_equivalence_range_seconds}")
    print(f"distribution={result.distribution} rms={result.rms_residual}")
    print(f"matched={len(result.matched_observations)} intervals={len(result.adopted_intervals)}")
    print(f"motion={result.per_batch_motion}")
    for reason in result.reasons:
        print(f"  reason: {reason}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    for note in result.identifiability_notes:
        print(f"  note: {note}")


def main():
    full_times = np.linspace(0.0, 4.0, 81)
    known = make_batch("known-motion", curved, full_times, seed=1)
    report(
        "known accelerated motion: offset and transform recovered",
        calibrate_temporal(CalibrationInput("ds", (known,)), CONVENTION, VERSION),
    )

    straight_batch = make_batch("straight-line", straight, full_times, seed=2)
    report(
        "uniform straight line: offset/phase indistinguishable",
        calibrate_temporal(CalibrationInput("ds", (straight_batch,)), CONVENTION, VERSION),
    )

    still_batch = make_batch("stationary", stationary, np.linspace(0.0, 2.0, 41), seed=3)
    report(
        "stationary target: offset unobservable",
        calibrate_temporal(CalibrationInput("ds", (still_batch,)), CONVENTION, VERSION),
    )

    short_batch = make_batch("short", curved, np.linspace(0.0, 0.08, 3), seed=4)
    report(
        "deliberately too-short record",
        calibrate_temporal(CalibrationInput("ds", (short_batch,)), CONVENTION, VERSION),
    )

    fixed = calibrate_temporal(
        CalibrationInput("ds", (known,)), CONVENTION, VERSION,
        TemporalCalibrationConfig(mode="fixed_offset", fixed_offset_seconds=TRUE_OFFSET),
    )
    simultaneous = calibrate_temporal(
        CalibrationInput("ds", (known,)), CONVENTION, VERSION,
        TemporalCalibrationConfig(mode="simultaneous_only", simultaneous_tolerance_seconds=0.002),
    )
    report("fixed offset mode", fixed)
    report("simultaneous correspondences only", simultaneous)

    gapped = make_batch(
        "gapped", curved, full_times, seed=5,
        missing=((1.5 - TRUE_OFFSET, 2.6 - TRUE_OFFSET),),
    )
    gapped_result = calibrate_temporal(CalibrationInput("ds", (gapped,)), CONVENTION, VERSION)
    report("recording gap is never crossed", gapped_result)
    print(f"gaps(raw units)={gapped_result.gaps}")

    millisecond = make_batch("known-ms", curved, full_times, seed=1, unit="ms")
    report(
        "millisecond clocks: identical physical answer",
        calibrate_temporal(CalibrationInput("ds", (millisecond,)), CONVENTION, VERSION),
    )

    helper = make_batch("helper", curved, full_times, seed=6)
    conflicting = make_batch("conflicting", curved, full_times, seed=7, offset=0.4)
    report(
        "two accelerated batches with incompatible offsets",
        calibrate_temporal(
            CalibrationInput("ds", (helper, conflicting)), CONVENTION, VERSION
        ),
    )

    full, comparisons = compare_temporal_scenarios(
        CalibrationInput("ds", (known, still_batch)),
        CONVENTION,
        VERSION,
        {
            "drop-stationary-batch": TemporalScenario(
                excluded_batches=frozenset({"stationary"})
            ),
            "drop-early-window": TemporalScenario(
                excluded_windows={"known-motion": ((0.0, 1.0),)}
            ),
        },
    )
    report("full multi-batch scenario", full)
    for comparison in comparisons:
        print(f"\nscenario {comparison.label}")
        print(f"offset_change_s={comparison.offset_change_seconds}")
        for basis in comparison.judgment_basis:
            print(f"  basis: {basis}")

    saved = full.as_dict()
    assert saved["matched_observations"], "saved result must trace adopted matches"
    assert saved["adopted_intervals"], "saved result must trace adopted intervals"
    raw_unchanged = all(
        observation.timestamp_b is not None
        for batch in (known, still_batch)
        for observation in batch.observations
    )
    assert raw_unchanged
    print("\ntemporal validation completed; raw observations unchanged and provenance saved")


if __name__ == "__main__":
    main()
