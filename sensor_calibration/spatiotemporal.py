from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .calibration import CalibrationConfig, _fit_transform
from .models import (
    BatchTimeSelection,
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    Observation,
    ObservationBatch,
    RigidTransform,
    SpatiotemporalComparison,
    SpatiotemporalResult,
    SpatiotemporalScenario,
    TimeInterval,
    TimeOffsetEstimate,
)


@dataclass(frozen=True)
class SpatiotemporalConfig:
    """Options for joint rigid-relation / fixed clock-offset calibration.

    Sign convention: t_b = t_a - offset_seconds.

    mode:
      joint_offset        - estimate one offset shared by all batches;
      fixed_offset        - use fixed_offset_seconds (no search);
      correspondable_only - use only observations matchable with zero offset
                            inside contiguous B segments (missing periods are
                            never bridged).
    A missing B period is any inter-sample gap larger than gap_factor times
    the median B sampling interval of that batch (factor is unit free).
    """

    base: CalibrationConfig = field(default_factory=CalibrationConfig)
    mode: str = "joint_offset"
    fixed_offset_seconds: float = 0.0
    search_half_range_seconds: float | None = None
    search_grid: int = 201
    refinement_rounds: int = 3
    gap_factor: float = 5.0
    min_pair_count: int = 4
    ambiguity_residual_factor: float = 4.0
    static_motion_factor: float = 8.0
    excluded_batches: frozenset[str] = frozenset()
    excluded_observations: frozenset[tuple[str, str]] = frozenset()
    excluded_a_windows: Mapping[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"joint_offset", "fixed_offset", "correspondable_only"}:
            raise ValueError(f"unknown spatiotemporal mode {self.mode!r}")
        object.__setattr__(self, "excluded_batches", frozenset(self.excluded_batches))
        object.__setattr__(self, "excluded_observations", frozenset(self.excluded_observations))
        object.__setattr__(
            self,
            "excluded_a_windows",
            {
                key: tuple(tuple(window) for window in windows)
                for key, windows in self.excluded_a_windows.items()
            },
        )


# A pair is one kept A sample plus the raw-clock bookkeeping needed to match
# it against the B stream under any candidate offset, so the same immutable
# records back every scenario and the raw observations are never rewritten.
#   0  batch_id
#   1  observation_id
#   2  sensor_a_position
#   3  position_sigma (within-batch weighting)
#   4  t_a in seconds
#   5  position_sigma (measurement noise scale reported on the result)
#   6  t_a in raw A-clock units
#   7  t_a in seconds (matching reference)
#   8  (b_lo_raw, b_hi_raw): span of the contiguous B segments, raw B clock
#   9  offset lower bound in seconds (match stays inside the B record)
#   10 offset upper bound in seconds
_Pair = tuple

# Per-batch segment registry used during one calibration call:
# batch_id -> list of (times_raw, clean_positions, noisy_positions, slopes,
#                     times_seconds, scale)
_SEGMENTS: dict[str, list[tuple]] = {}


def _batch_fingerprint(batch: ObservationBatch) -> str:
    digest = hashlib.sha256()
    digest.update(batch.sensor_a_id.encode())
    digest.update(batch.sensor_b_id.encode())
    digest.update(batch.time_unit.encode())
    for observation in sorted(batch.observations, key=lambda item: item.observation_id):
        digest.update(observation.observation_id.encode())
        digest.update(np.ascontiguousarray(observation.sensor_a_position).tobytes())
        digest.update(np.ascontiguousarray(observation.sensor_b_position).tobytes())
        digest.update(repr((observation.t_a, observation.t_b)).encode())
    return digest.hexdigest()


def _b_segments(
    batch: ObservationBatch, gap_factor: float
) -> tuple[list[list[Observation]], list[tuple[float, float]]]:
    """Split the B stream at missing periods and clock resets.

    Observations are walked in acquisition (A-clock) order.  A backward jump
    of the B clock is a reset: it starts a new segment and the ambiguous
    B-clock overlap is recorded as a gap span so the pairing provenance
    stays auditable.  Segments hold B observations ordered by raw B-clock
    time; gaps are raw-unit (start, end) spans that are never bridged."""
    ordered = sorted(batch.observations, key=lambda obs: obs.t_a)
    if len(ordered) < 2:
        return ([list(ordered)] if ordered else []), []
    times = np.array([obs.t_b for obs in ordered], dtype=float)
    steps = np.diff(times)
    positive_steps = steps[steps > 0]
    median_step = float(np.median(positive_steps)) if len(positive_steps) else 0.0
    segments: list[list[Observation]] = [[ordered[0]]]
    gaps: list[tuple[float, float]] = []
    for index, step in enumerate(steps, start=1):
        if step < 0:
            # Clock reset: the same B-clock readings now mean a different
            # physical period, so the two sides must never share a chord.
            gaps.append((float(times[index]), float(times[index - 1])))
            segments.append([ordered[index]])
        elif median_step > 0 and float(step) > gap_factor * median_step:
            gaps.append((float(times[index - 1]), float(times[index])))
            segments.append([ordered[index]])
        else:
            segments[-1].append(ordered[index])
    return segments, gaps


def _interpolate_segment(
    segment: list[Observation], scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    times_raw = np.array([obs.t_b for obs in segment], dtype=float)
    positions = np.vstack([
        obs.sensor_b_position_clean if obs.sensor_b_position_clean is not None
        else obs.sensor_b_position
        for obs in segment
    ])
    noisy_positions = np.vstack([obs.sensor_b_position for obs in segment])
    times_seconds = times_raw * scale
    # Velocity on each *piecewise-linear* interval [i-1, i].  The match under
    # an offset shift moves along this same chord, so the chord slope is the
    # exact re-interpolation velocity (a centred finite difference would not
    # match the linear interpolation and would bias the offset search).
    velocities = np.zeros_like(positions)
    if len(segment) >= 2:
        slopes = np.diff(positions, axis=0) / np.diff(times_seconds)[:, None]
        velocities[1:] = slopes
        velocities[0] = slopes[0]
    return times_raw, positions, noisy_positions, velocities, times_seconds, scale


def _segment_interp(
    segments_data,
    target_b_raw: float,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Piecewise-linear clean and noisy B position at a raw B time, strictly
    inside one contiguous segment (endpoints inclusive).  Missing periods
    are never bridged.  Returns clean_position, noisy_position, seg_lo,
    seg_hi in raw B-clock units."""
    for times_raw, clean, noisy, slopes, times_seconds, scale in segments_data:
        if target_b_raw < times_raw[0] or target_b_raw > times_raw[-1]:
            continue
        insert = int(np.searchsorted(times_raw, target_b_raw))
        if insert == 0:
            if target_b_raw == times_raw[0]:
                return clean[0].copy(), noisy[0].copy(), float(times_raw[0]), float(times_raw[-1])
            return None
        if target_b_raw == times_raw[-1]:
            return clean[-1].copy(), noisy[-1].copy(), float(times_raw[0]), float(times_raw[-1])
        if insert >= len(times_raw):
            return None
        if target_b_raw == times_raw[insert]:
            return clean[insert].copy(), noisy[insert].copy(), float(times_raw[0]), float(times_raw[-1])
        ratio = (target_b_raw - times_raw[insert - 1]) / (times_raw[insert] - times_raw[insert - 1])
        clean_position = (1.0 - ratio) * clean[insert - 1] + ratio * clean[insert]
        noisy_position = (1.0 - ratio) * noisy[insert - 1] + ratio * noisy[insert]
        return clean_position, noisy_position, float(times_raw[0]), float(times_raw[-1])
    return None


def _match_pair(pair: _Pair, offset_seconds: float):
    """Look up the B reading that matches one A sample under the offset.

    t_b = t_a - offset (seconds); interpolation is confined to the contiguous
    segment containing that time.  Returns (clean_b, noisy_b, chord_slope,
    chord_lo_raw, chord_hi_raw) or None if the match would fall in a gap or
    outside the B record."""
    batch_id = pair[0]
    t_a_seconds = pair[4]
    target_b_seconds = t_a_seconds - offset_seconds
    for times_raw, clean, noisy, slopes, times_seconds, scale in _SEGMENTS.get(batch_id, ()):
        target_raw = target_b_seconds / scale
        if target_raw < times_raw[0] or target_raw > times_raw[-1]:
            continue
        insert = int(np.searchsorted(times_raw, target_raw))
        if insert == 0:
            if target_raw == times_raw[0] and len(times_raw) > 1:
                return clean[0].copy(), noisy[0].copy(), slopes[0], float(times_raw[0]), float(times_raw[1])
            return None
        if target_raw == times_raw[-1]:
            return clean[-1].copy(), noisy[-1].copy(), slopes[-1], float(times_raw[-2]), float(times_raw[-1])
        if insert >= len(times_raw):
            return None
        if target_raw == times_raw[insert]:
            return clean[insert].copy(), noisy[insert].copy(), slopes[insert], float(times_raw[insert - 1]), float(times_raw[insert])
        ratio = (target_raw - times_raw[insert - 1]) / (times_raw[insert] - times_raw[insert - 1])
        clean_position = (1.0 - ratio) * clean[insert - 1] + ratio * clean[insert]
        noisy_position = (1.0 - ratio) * noisy[insert - 1] + ratio * noisy[insert]
        return (
            clean_position,
            noisy_position,
            slopes[insert],
            float(times_raw[insert - 1]),
            float(times_raw[insert]),
        )
    return None


def _build_pairs(
    batch: ObservationBatch,
    config: SpatiotemporalConfig,
    segments_data: list[tuple],
    windows_raw: tuple[tuple[float, float], ...],
) -> tuple[
    list[_Pair],
    dict[str, str],
    list[tuple[float, float]],
    list[tuple[float, float]],
    tuple[float, float],
]:
    """Select the A samples allowed to participate and bind each one to the
    B record it may match against.

    The contiguous B segments set the feasible offset bracket of every pair
    (a match may never be interpolated across a missing period), and the
    excluded A-clock windows remove observations here, before any offset
    search.  Matching itself is offset dependent and happens through
    _match_pair.  Returns the pairs, the omitted observations with reasons,
    the used raw A-clock intervals, the raw B-clock interval each pair can
    match inside, and the batch-level feasible offset range."""
    scale = batch.seconds_per_unit
    # Register the segments so _match_pair can resolve matches for this
    # batch even outside a full calibrate_spatiotemporal call.
    _SEGMENTS[batch.batch_id] = segments_data
    if segments_data:
        b_lo_seconds = min(float(segment[4][0]) for segment in segments_data)
        b_hi_seconds = max(float(segment[4][-1]) for segment in segments_data)
    else:
        b_lo_seconds = b_hi_seconds = 0.0
    pairs: list[_Pair] = []
    omitted: dict[str, str] = {}
    for observation in sorted(batch.observations, key=lambda obs: obs.t_a):
        reference = (batch.batch_id, observation.observation_id)
        if config.base.exclude_marked_bad and observation.quality == "bad":
            omitted[observation.observation_id] = "marked_quality_bad"
        elif reference in config.excluded_observations:
            omitted[observation.observation_id] = "excluded_by_scenario"
        elif any(lo <= observation.t_a <= hi for lo, hi in windows_raw):
            omitted[observation.observation_id] = "excluded_time_window"
        else:
            t_a_raw = float(observation.t_a)
            t_a_seconds = t_a_raw * scale
            # t_b = t_a - offset must stay inside the B record.
            offset_lo = t_a_seconds - b_hi_seconds
            offset_hi = t_a_seconds - b_lo_seconds
            pairs.append((
                batch.batch_id,
                observation.observation_id,
                observation.sensor_a_position,
                observation.position_sigma,
                t_a_seconds,
                observation.position_sigma,
                t_a_raw,
                t_a_seconds,
                (b_lo_seconds / scale, b_hi_seconds / scale),
                offset_lo,
                offset_hi,
            ))
    a_intervals_raw = [(pair[6], pair[6]) for pair in pairs]
    # Raw B-clock spans of the contiguous segments backing the pairs;
    # missing periods and reset overlaps stay visible as separate intervals.
    b_hits = [
        (float(segment[0][0]), float(segment[0][-1]))
        for segment in segments_data
    ]
    if pairs:
        feasible = (
            float(min(pair[9] for pair in pairs)),
            float(max(pair[10] for pair in pairs)),
        )
    else:
        feasible = (0.0, 0.0)
    return pairs, omitted, a_intervals_raw, b_hits, feasible


def _active_mask(pairs: list[_Pair], offset_seconds: float) -> np.ndarray:
    """Participation mask at one candidate offset: the per-pair feasible
    bracket must contain the offset *and* the shifted match must land inside
    a contiguous B segment (missing periods are never bridged)."""
    return np.array([
        pair[9] <= offset_seconds <= pair[10]
        and _match_pair(pair, offset_seconds) is not None
        for pair in pairs
    ], dtype=bool)


def _equal_share_weights(
    pairs: list[_Pair], duplicate_size: Mapping[str, int], active: np.ndarray | None = None
) -> np.ndarray:
    by_batch: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        if active is not None and not active[index]:
            continue
        by_batch.setdefault(pair[0], []).append(index)
    weights = np.zeros(len(pairs), dtype=float)
    for batch_id, indices in by_batch.items():
        share = 1.0 / duplicate_size[batch_id]
        sigmas = np.array([pairs[i][3] if pairs[i][3] is not None else 1.0 for i in indices])
        within = (1.0 / np.square(sigmas))
        within = within / float(np.sum(within))
        for position, index in enumerate(indices):
            weights[index] = share * within[position]
    return weights


def _resolve_matches(
    pairs: list[_Pair], offset_seconds: float
) -> tuple[list[_Pair], np.ndarray, np.ndarray, np.ndarray]:
    """Resolve B matches for every pair at the candidate offset.

    Returns the participating pairs (batch equal share is applied over these)
    plus A points, measured B points at the matched time and the clean chord
    velocities (per second).  Temporal re-interpolation uses the clean chord;
    residuals are checked against the noisy coordinate at that same time, so
    measurement noise is neither stretched nor rewritten."""
    active_pairs: list[_Pair] = []
    points_a_list: list[np.ndarray] = []
    points_b_list: list[np.ndarray] = []
    velocities_list: list[np.ndarray] = []
    for pair in pairs:
        matched = _match_pair(pair, offset_seconds)
        if matched is None:
            continue
        _clean_position, noisy_position, slope, _chord_lo, _chord_hi = matched
        active_pairs.append(pair)
        points_a_list.append(pair[2])
        points_b_list.append(noisy_position)
        velocities_list.append(slope)
    points_a = np.vstack(points_a_list) if points_a_list else np.empty((0, 3))
    points_b = np.vstack(points_b_list) if points_b_list else np.empty((0, 3))
    velocities = np.vstack(velocities_list) if velocities_list else np.empty((0, 3))
    return active_pairs, points_a, points_b, velocities


# Populated per calibrate call so batch shares can be normalised.
_DUP_SIZES: dict[str, float] = {}


def _evaluate(
    pairs: list[_Pair],
    offset_seconds: float,
    active: np.ndarray | None = None,
) -> tuple[
    RigidTransform | None,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fit the rigid transform at one candidate offset.

    Returns the transform, the weighted RMS residual and, aligned with the
    full ``pairs`` list, the A points, the matched noisy B points, the
    per-pair weights and the clean chord velocities (per second).  Rows of
    inactive pairs are zero so callers can index with the same mask."""
    if active is None:
        active = _active_mask(pairs, offset_seconds)
    active = np.asarray(active, dtype=bool)
    count = len(pairs)
    points_a = np.zeros((count, 3))
    points_b = np.zeros((count, 3))
    weights = np.zeros(count)
    velocities = np.zeros((count, 3))
    chosen: list[int] = []
    for index, pair in enumerate(pairs):
        if not active[index]:
            continue
        matched = _match_pair(pair, offset_seconds)
        if matched is None:
            continue
        _clean_position, noisy_position, slope, _chord_lo, _chord_hi = matched
        chosen.append(index)
        points_a[index] = pair[2]
        points_b[index] = noisy_position
        velocities[index] = slope
    if not chosen:
        return None, float("inf"), points_a, points_b, weights, velocities
    chosen_array = np.array(chosen, dtype=int)
    sub_weights = _equal_share_weights([pairs[i] for i in chosen], _DUP_SIZES)
    rigid, *_ = _fit_transform(points_a[chosen_array], points_b[chosen_array], sub_weights)
    if rigid is None:
        return None, float("inf"), points_a, points_b, weights, velocities
    residuals = np.linalg.norm(
        rigid.apply(points_a[chosen_array]) - points_b[chosen_array], axis=1
    )
    normalized = sub_weights * len(chosen) / float(np.sum(sub_weights))
    rms = float(np.sqrt(np.sum(normalized * residuals**2) / len(chosen)))
    weights[chosen_array] = sub_weights
    return rigid, rms, points_a, points_b, weights, velocities


def _search_offset(
    pairs: list[_Pair],
    feasible_range: tuple[float, float],
    config: SpatiotemporalConfig,
) -> tuple[float, float]:
    lo, hi = feasible_range
    if config.search_half_range_seconds is not None:
        half = abs(config.search_half_range_seconds)
        lo, hi = max(lo, -half), min(hi, half)
    if hi < lo:
        raise ValueError("no feasible offset inside the configured search range")

    def objective(offset: float) -> float:
        active_pairs_local, _pa, _pb, _vel = _resolve_matches(pairs, offset)
        if len(active_pairs_local) < config.min_pair_count:
            return float("inf")
        _, rms, *_ = _evaluate(pairs, offset)
        return rms if np.isfinite(rms) else float("inf")

    grid = np.linspace(lo, hi, config.search_grid)
    values = [objective(float(value)) for value in grid]
    best_index = int(np.argmin(values))
    best_offset = float(grid[best_index])
    for _ in range(config.refinement_rounds):
        left = float(grid[max(best_index - 1, 0)])
        right = float(grid[min(best_index + 1, len(grid) - 1)])
        grid = np.linspace(left, right, config.search_grid)
        values = [objective(float(value)) for value in grid]
        best_index = int(np.argmin(values))
        best_offset = float(grid[best_index])
    return best_offset, objective(best_offset)


def _ambiguity_profile_range(
    pairs: list[_Pair],
    feasible_range: tuple[float, float],
    minimum_rms: float,
    tolerance: float,
    config: SpatiotemporalConfig,
) -> tuple[float, float]:
    """Largest contiguous offset interval around the minimum whose weighted
    RMS stays within minimum_rms + tolerance.  This is the family of offset
    values that cannot be told apart from the fitted one at the data's noise
    level - the honest answer for constant-velocity straight motion."""
    lo, hi = feasible_range
    grid = np.linspace(lo, hi, config.search_grid)
    acceptable: list[float] = []
    for value in grid:
        active_pairs_local, _pa, _pb, _vel = _resolve_matches(pairs, float(value))
        if len(active_pairs_local) < config.min_pair_count:
            continue
        _, rms, *_ = _evaluate(pairs, float(value))
        if rms <= minimum_rms + tolerance:
            acceptable.append(float(value))
    if not acceptable:
        return feasible_range
    return acceptable[0], acceptable[-1]


def _spatial_distribution(points_a: np.ndarray, points_b: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    scale_a = np.linalg.svd(points_a - points_a.mean(axis=0), compute_uv=False)
    scale_b = np.linalg.svd(points_b - points_b.mean(axis=0), compute_uv=False)
    rank_a = int(np.count_nonzero(scale_a > 1e-9 * max(scale_a[0], np.finfo(float).tiny)))
    rank_b = int(np.count_nonzero(scale_b > 1e-9 * max(scale_b[0], np.finfo(float).tiny)))
    rank = min(rank_a, rank_b)
    return {0: "point", 1: "line", 2: "plane", 3: "volume"}[rank], scale_a, scale_b


def _identifiability(
    active_pairs: list[_Pair],
    points_a: np.ndarray,
    velocities: np.ndarray,
    rigid: RigidTransform,
    points_b: np.ndarray,
    weights: np.ndarray,
    measurement_sigma: float,
    config: SpatiotemporalConfig,
) -> TimeOffsetEstimate:
    """Linearised information about the offset after eliminating rotation
    (3 axis-angle components) and translation (3 components).

    Residual r = R p_a + t - (p_b0 + v_b dt), so dr/ddt = -v_b and
    dr/dt = -skew(R p_a) for the rotation part.  The Schur complement of the
    6x6 rigid block is the information carried about dt beyond any rigid
    adjustment.  Zero information means another (dt, transform) explanation
    fits identically, and a small residual must never be reported as a
    unique calibration."""
    rotated = points_a @ rigid.rotation.T
    w = weights * len(weights) / float(np.sum(weights))
    # Jacobian columns: translation x,y,z (identity) then rotation via the
    # skew-symmetric generator d(R p)/d(alpha) = -skew(R p).
    n = len(active_pairs)
    jacobian = np.zeros((3 * n, 7))
    for row, point in enumerate(rotated):
        base = 3 * row
        jacobian[base:base + 3, 0:3] = np.eye(3)
        x, y, z = point
        jacobian[base:base + 3, 3] = np.array([0.0, z, -y])
        jacobian[base:base + 3, 4] = np.array([-z, 0.0, x])
        jacobian[base:base + 3, 5] = np.array([y, -x, 0.0])
        jacobian[base:base + 3, 6] = -velocities[row]
    full_weight = np.repeat(w, 3) / (measurement_sigma**2 if measurement_sigma > 0 else 1.0)
    information = jacobian.T @ (full_weight[:, None] * jacobian)
    rigid_block = information[:6, :6]
    cross = information[:6, 6]
    raw_offset_information = float(information[6, 6])
    offset_information = float(information[6, 6] - cross @ np.linalg.pinv(rigid_block) @ cross)
    information_ratio = (
        offset_information / raw_offset_information if raw_offset_information > 0 else 0.0
    )

    speeds = np.linalg.norm(velocities, axis=1)
    median_speed = float(np.median(speeds))
    points_b_used = points_b
    scatter = np.linalg.svd(points_b_used - points_b_used.mean(axis=0), compute_uv=False)
    motion_scale = float(scatter[0])
    static_floor = config.static_motion_factor * measurement_sigma
    # Null direction of the rigid block projected dt sensitivity: any part of
    # v_b that a translation/rotation can absorb.
    if measurement_sigma <= 0:
        sigma_offset = None
    else:
        sigma_offset = float(1.0 / np.sqrt(offset_information)) if offset_information > 0 else None

    notes: list[str] = []
    ambiguity_direction = None
    equivalent = None
    if motion_scale <= static_floor:
        status = "unobservable_target_static"
        notes.append("target motion is below the measurement-noise floor; offset leaves no residual signature")
    elif information_ratio < 1e-6:
        status = "confounded_with_motion"
        mean_velocity = np.average(velocities, axis=0, weights=w)
        norm = float(np.linalg.norm(mean_velocity))
        ambiguity_direction = mean_velocity / norm if norm > 0 else np.zeros(3)
        equivalent = ambiguity_direction.copy()
        notes.append(
            "constant-velocity straight motion: changing the offset by dt is "
            "absorbed by a translation change of -v*dt; many (offset, translation) pairs fit"
        )
    else:
        status = "identifiable"
        notes.append("velocity changes (curvature/acceleration or non-parallel batch motions) pin the fixed offset")
    return TimeOffsetEstimate(
        offset_seconds=None,
        status=status,
        sigma_seconds=sigma_offset,
        feasible_offset_range_seconds=None,
        ambiguity_direction=ambiguity_direction,
        equivalent_translation_per_second=equivalent,
        notes=tuple(notes),
    )


def _merge_intervals(
    intervals: list[tuple[float, float]], clock: str, ids_by_time: dict[float, list[str]] | None = None
) -> tuple[TimeInterval, ...]:
    if not intervals:
        return ()
    ordered = sorted(intervals)
    merged: list[list[float]] = []
    for lo, hi in ordered:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return tuple(TimeInterval(float(lo), float(hi), clock) for lo, hi in merged)


def calibrate_spatiotemporal(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    config: SpatiotemporalConfig | None = None,
) -> SpatiotemporalResult:
    """Jointly estimate the rigid relation and one fixed clock offset.

    Raw timestamps are never rewritten: each returned BatchTimeSelection
    stores the raw A- and B-clock intervals actually used, the B gaps that
    were not bridged, and every omitted A observation with a reason.
    """
    global _DUP_SIZES, _SEGMENTS
    config = config or SpatiotemporalConfig()

    active_batches = tuple(
        sorted(
            (
                batch
                for batch in calibration_input.batches
                if batch.batch_id not in config.excluded_batches
            ),
            key=lambda batch: batch.batch_id,
        )
    )
    untimed = [batch.batch_id for batch in active_batches if not batch.is_timed()]
    if untimed:
        return SpatiotemporalResult(
            success=False,
            mode=config.mode,
            transform=None,
            time_offset=TimeOffsetEstimate(None, "not_estimated", notes=()),
            coordinate_convention=convention,
            calibration_version=version,
            selection=(),
            used_observations=(),
            excluded_observations=(),
            reasons=(f"batches_without_paired_timestamps:{','.join(sorted(untimed))}",),
        )

    fingerprints = {batch.batch_id: _batch_fingerprint(batch) for batch in active_batches}
    group_size: dict[str, int] = {}
    batch_dup_size: dict[str, float] = {}
    for fingerprint in fingerprints.values():
        group_size[fingerprint] = group_size.get(fingerprint, 0) + 1
    for batch_id, fingerprint in fingerprints.items():
        batch_dup_size[batch_id] = float(group_size[fingerprint])
    _DUP_SIZES = batch_dup_size
    duplicate_groups = tuple(
        tuple(sorted(bid for bid, fp in fingerprints.items() if fp == fingerprint))
        for fingerprint, count in group_size.items()
        if count > 1
    )

    all_pairs: list[_Pair] = []
    selections: list[BatchTimeSelection] = []
    excluded_refs: list[tuple[str, str]] = []
    batch_feasible: dict[str, tuple[float, float]] = {}
    segment_registry: dict[str, list[tuple]] = {}
    for batch in active_batches:
        segments, gaps = _b_segments(batch, config.gap_factor)
        segments_data = [_interpolate_segment(segment, batch.seconds_per_unit) for segment in segments]
        segment_registry[batch.batch_id] = segments_data
        windows_raw = tuple(
            (lo / batch.seconds_per_unit, hi / batch.seconds_per_unit)
            for lo, hi in config.excluded_a_windows.get(batch.batch_id, ())
        )
        pairs, omitted, a_intervals_raw, b_hits, feasible = _build_pairs(
            batch, config, segments_data, windows_raw
        )
        batch_feasible[batch.batch_id] = feasible
        all_pairs.extend(pairs)
        excluded_refs.extend(
            (batch.batch_id, obs_id)
            for obs_id in omitted
        )
        active_ids = {pair[1] for pair in pairs}
        selections.append(
            BatchTimeSelection(
                batch_id=batch.batch_id,
                time_unit=batch.time_unit,
                used_a_observations=tuple(sorted(active_ids)),
                used_intervals_a_clock=_merge_intervals(a_intervals_raw, "a"),
                used_intervals_b_clock=_merge_intervals(b_hits, "b"),
                gaps_b_clock=tuple(gaps),
                omitted_a_observations=tuple(sorted(omitted)),
                omitted_reasons=omitted,
                feasible_offset_range=(None if not pairs else feasible),
                paired_count=len(pairs),
            )
        )

    _SEGMENTS = segment_registry

    # Union over pairs: different offsets activate different chords.  The
    # objective itself returns infinity where too few pairs are active.
    feasible_pairs = [pair for pair in all_pairs if pair[9] <= pair[10]]
    global_range = (
        float(min(pair[9] for pair in feasible_pairs)),
        float(max(pair[10] for pair in feasible_pairs)),
    ) if feasible_pairs else (0.0, -1.0)

    notes: list[str] = []
    if config.mode == "fixed_offset":
        chosen_offset = float(config.fixed_offset_seconds)
        active = _active_mask(all_pairs, chosen_offset)
        offset_status = "fixed_offset_provided"
    elif config.mode == "correspondable_only":
        chosen_offset = 0.0
        active = _active_mask(all_pairs, 0.0)
        offset_status = "not_estimated"
        notes.append("only observations correspondable at zero offset without bridging gaps are used")
    else:
        if global_range[1] < global_range[0]:
            return SpatiotemporalResult(
                success=False,
                mode=config.mode,
                transform=None,
                time_offset=TimeOffsetEstimate(None, "not_estimated"),
                coordinate_convention=convention,
                calibration_version=version,
                selection=tuple(selections),
                used_observations=(),
                excluded_observations=tuple(excluded_refs),
                reasons=("no_common_offset_interval_across_batches",),
                duplicate_groups=duplicate_groups,
                effective_batch_count=float(len(group_size)),
            )
        chosen_offset, _ = _search_offset(all_pairs, global_range, config)
        active = _active_mask(all_pairs, chosen_offset)
        offset_status = None

    if int(np.count_nonzero(active)) < config.min_pair_count:
        return SpatiotemporalResult(
            success=False,
            mode=config.mode,
            transform=None,
            time_offset=TimeOffsetEstimate(
                chosen_offset if offset_status is None else chosen_offset,
                offset_status or "not_estimated",
            ),
            coordinate_convention=convention,
            calibration_version=version,
            selection=tuple(selections),
            used_observations=(),
            excluded_observations=tuple(excluded_refs),
            reasons=(f"at_least_{config.min_pair_count}_correspondable_pairs_required",),
            duplicate_groups=duplicate_groups,
            effective_batch_count=float(len(group_size)),
        )

    rigid, rms, points_a_all, points_b_all, weights_all, velocities_all = _evaluate(
        all_pairs, chosen_offset, active
    )
    used_refs = tuple(
        (pair[0], pair[1]) for pair, flag in zip(all_pairs, active) if flag
    )
    active_indices = np.flatnonzero(active)
    sigmas = np.array(
        [all_pairs[i][5] for i in active_indices if all_pairs[i][5] is not None], dtype=float
    )
    measurement_sigma = float(np.max(sigmas)) if len(sigmas) else 0.0
    distribution, _, _ = _spatial_distribution(
        points_a_all[active_indices], points_b_all[active_indices]
    )

    # Record which pairs really participate at the chosen offset on each batch.
    final_selections: list[BatchTimeSelection] = []
    participating = {}
    for pair, flag in zip(all_pairs, active):
        if flag:
            participating.setdefault(pair[0], set()).add(pair[1])
    for selection in selections:
        kept = participating.get(selection.batch_id, set())
        reasons = dict(selection.omitted_reasons)
        for obs_id in selection.used_a_observations:
            if obs_id not in kept:
                reasons[obs_id] = "outside_contiguous_segment_at_chosen_offset"
        final_selections.append(
            BatchTimeSelection(
                batch_id=selection.batch_id,
                time_unit=selection.time_unit,
                used_a_observations=tuple(sorted(kept)),
                used_intervals_a_clock=selection.used_intervals_a_clock,
                used_intervals_b_clock=selection.used_intervals_b_clock,
                gaps_b_clock=selection.gaps_b_clock,
                omitted_a_observations=tuple(sorted(reasons)),
                omitted_reasons=reasons,
                feasible_offset_range=selection.feasible_offset_range,
                paired_count=len(kept),
            )
        )

    spatial_status = "valid" if rigid is not None and distribution in {"plane", "volume"} else "unstable"
    if rigid is None:
        return SpatiotemporalResult(
            success=False,
            mode=config.mode,
            transform=None,
            time_offset=TimeOffsetEstimate(chosen_offset, offset_status or "not_estimated"),
            coordinate_convention=convention,
            calibration_version=version,
            selection=tuple(final_selections),
            used_observations=used_refs,
            excluded_observations=tuple(excluded_refs),
            rms_residual=None,
            measurement_sigma=measurement_sigma,
            spatial_distribution=distribution,
            spatial_status="undetermined",
            reasons=("only_a_reflection_can_fit_the_correspondences",),
            duplicate_groups=duplicate_groups,
            effective_batch_count=float(len(group_size)),
        )

    rigid = RigidTransform(rigid.rotation, rigid.translation, convention.frame_a, convention.frame_b)
    if config.mode == "joint_offset":
        estimate = _identifiability(
            [all_pairs[i] for i in active_indices],
            points_a_all[active_indices],
            velocities_all[active_indices],
            rigid,
            points_b_all[active_indices],
            weights_all[active_indices],
            measurement_sigma,
            config,
        )
        # A static target carries no offset information at all: the searched
        # number is just where the flat objective happened to stop.
        if estimate.status == "unobservable_target_static":
            object.__setattr__(estimate, "offset_seconds", None)
            object.__setattr__(estimate, "sigma_seconds", None)
            object.__setattr__(estimate, "feasible_offset_range_seconds", global_range)
        else:
            object.__setattr__(estimate, "offset_seconds", chosen_offset)
        if estimate.status == "identifiable" and estimate.sigma_seconds is not None:
            reach = config.ambiguity_residual_factor * estimate.sigma_seconds
            object.__setattr__(
                estimate,
                "feasible_offset_range_seconds",
                (chosen_offset - reach, chosen_offset + reach),
            )
        elif estimate.status == "confounded_with_motion":
            tolerance = config.ambiguity_residual_factor * max(measurement_sigma, 1e-12)
            profile_range = _ambiguity_profile_range(
                all_pairs, global_range, rms, tolerance, config
            )
            object.__setattr__(estimate, "feasible_offset_range_seconds", profile_range)
            object.__setattr__(estimate, "sigma_seconds", None)
        notes.extend(estimate.notes)
    else:
        estimate = TimeOffsetEstimate(chosen_offset, offset_status, notes=tuple(notes))

    if distribution in {"point", "line"}:
        spatial_status = "undetermined"
        notes.append(f"merged spatial geometry is {distribution}; the rigid transform is not fully determined")
    for group in duplicate_groups:
        notes.append(
            "batches " + ", ".join(f"'{name}'" for name in group)
            + " are identical acquisitions; they share one acquisition weight"
        )
    residual_consistent = True
    if (
        measurement_sigma > 0
        and np.isfinite(rms)
        and rms > config.ambiguity_residual_factor * measurement_sigma
    ):
        # The best (offset, transform) explanation still misfits the reported
        # noise by a wide margin: the pairings are mutually inconsistent (for
        # example a clock reset mid-stream).  Reporting a clean success here
        # would be an arbitrary transform, so the result is downgraded.
        residual_consistent = False
        if spatial_status == "valid":
            spatial_status = "unstable"
        notes.append(
            "weighted residuals exceed the reported measurement noise; no single "
            "offset and transform explain the pairings (possible clock reset or drift)"
        )
    success = rigid is not None and spatial_status != "undetermined" and residual_consistent

    return SpatiotemporalResult(
        success=success,
        mode=config.mode,
        transform=rigid,
        time_offset=estimate,
        coordinate_convention=convention,
        calibration_version=version,
        selection=tuple(final_selections),
        used_observations=used_refs,
        excluded_observations=tuple(excluded_refs),
        rms_residual=rms,
        measurement_sigma=measurement_sigma,
        spatial_distribution=distribution,
        spatial_status=spatial_status,
        identifiability_notes=tuple(notes),
        duplicate_groups=duplicate_groups,
        effective_batch_count=float(len(group_size)),
    )


def _transform_changes(reference: SpatiotemporalResult, candidate: SpatiotemporalResult):
    rotation_change = None
    translation_change = None
    if reference.transform is not None and candidate.transform is not None:
        rotation_change = float(
            np.linalg.norm(candidate.transform.rotation - reference.transform.rotation, ord="fro")
        )
        translation_change = float(
            np.linalg.norm(candidate.transform.translation - reference.transform.translation)
        )
    return rotation_change, translation_change


def compare_spatiotemporal_approaches(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    fixed_offset_seconds: float = 0.0,
    config: SpatiotemporalConfig | None = None,
) -> tuple[SpatiotemporalResult, tuple[SpatiotemporalComparison, ...]]:
    """Compare the three admissible ways of treating time:

    1. fixed_offset          - offset pinned to a user-supplied value;
    2. joint_offset          - R, t and offset estimated together;
    3. correspondable_only   - only zero-offset correspondable pairs used.
    Every result keeps the raw timestamps and the per-batch used intervals.
    """
    base = config or SpatiotemporalConfig()
    fixed_config = SpatiotemporalConfig(
        base=base.base, mode="fixed_offset",
        fixed_offset_seconds=fixed_offset_seconds,
        search_half_range_seconds=base.search_half_range_seconds,
        search_grid=base.search_grid, refinement_rounds=base.refinement_rounds,
        gap_factor=base.gap_factor, min_pair_count=base.min_pair_count,
        ambiguity_residual_factor=base.ambiguity_residual_factor,
        static_motion_factor=base.static_motion_factor,
        excluded_batches=base.excluded_batches,
        excluded_observations=base.excluded_observations,
        excluded_a_windows=base.excluded_a_windows,
    )
    joint_config = SpatiotemporalConfig(
        base=base.base, mode="joint_offset",
        fixed_offset_seconds=base.fixed_offset_seconds,
        search_half_range_seconds=base.search_half_range_seconds,
        search_grid=base.search_grid, refinement_rounds=base.refinement_rounds,
        gap_factor=base.gap_factor, min_pair_count=base.min_pair_count,
        ambiguity_residual_factor=base.ambiguity_residual_factor,
        static_motion_factor=base.static_motion_factor,
        excluded_batches=base.excluded_batches,
        excluded_observations=base.excluded_observations,
        excluded_a_windows=base.excluded_a_windows,
    )
    overlap_config = SpatiotemporalConfig(
        base=base.base, mode="correspondable_only",
        fixed_offset_seconds=base.fixed_offset_seconds,
        search_half_range_seconds=base.search_half_range_seconds,
        search_grid=base.search_grid, refinement_rounds=base.refinement_rounds,
        gap_factor=base.gap_factor, min_pair_count=base.min_pair_count,
        ambiguity_residual_factor=base.ambiguity_residual_factor,
        static_motion_factor=base.static_motion_factor,
        excluded_batches=base.excluded_batches,
        excluded_observations=base.excluded_observations,
        excluded_a_windows=base.excluded_a_windows,
    )
    fixed_result = calibrate_spatiotemporal(calibration_input, convention, version, fixed_config)
    joint_result = calibrate_spatiotemporal(calibration_input, convention, version, joint_config)
    overlap_result = calibrate_spatiotemporal(calibration_input, convention, version, overlap_config)

    comparisons: list[SpatiotemporalComparison] = []
    for label, result in (
        (f"fixed_offset@{fixed_offset_seconds:g}s", fixed_result),
        ("joint_offset_estimate", joint_result),
        ("correspondable_only", overlap_result),
    ):
        rotation_change, translation_change = _transform_changes(joint_result, result)
        offset_change = None
        if result.time_offset.offset_seconds is not None and joint_result.time_offset.offset_seconds is not None:
            offset_change = float(result.time_offset.offset_seconds - joint_result.time_offset.offset_seconds)
        basis = [
            f"mode:{result.mode}",
            f"spatial_status:{result.spatial_status}",
            f"time_offset_status:{result.time_offset.status}",
            f"used_pair_count:{len(result.used_observations)}",
            f"joint_used_pair_count:{len(joint_result.used_observations)}",
            f"rms_residual:{result.rms_residual if result.rms_residual is not None else 'none'}",
        ]
        if result.time_offset.status == "confounded_with_motion":
            basis.append("offset_not_unique_translation_can_absorb_it")
        if result.time_offset.status == "unobservable_target_static":
            basis.append("offset_unobservable_for_static_target")
        comparisons.append(
            SpatiotemporalComparison(
                label=label,
                mode=result.mode,
                result=result,
                offset_change_seconds=offset_change,
                rotation_change=rotation_change,
                translation_change=translation_change,
                judgment_basis=tuple(basis),
            )
        )
    return joint_result, tuple(comparisons)


def compare_spatiotemporal_scenarios(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    scenarios: Mapping[str, SpatiotemporalScenario],
    config: SpatiotemporalConfig | None = None,
) -> tuple[SpatiotemporalResult, tuple[SpatiotemporalComparison, ...]]:
    """Re-run the joint estimate after excluding whole batches, single
    observations or local A-clock windows.  Source data is never modified."""
    base = config or SpatiotemporalConfig()
    full = calibrate_spatiotemporal(calibration_input, convention, version, base)
    comparisons: list[SpatiotemporalComparison] = []
    for label, scenario in scenarios.items():
        scenario_config = SpatiotemporalConfig(
            base=base.base,
            mode=base.mode,
            fixed_offset_seconds=base.fixed_offset_seconds,
            search_half_range_seconds=base.search_half_range_seconds,
            search_grid=base.search_grid,
            refinement_rounds=base.refinement_rounds,
            gap_factor=base.gap_factor,
            min_pair_count=base.min_pair_count,
            ambiguity_residual_factor=base.ambiguity_residual_factor,
            static_motion_factor=base.static_motion_factor,
            excluded_batches=frozenset(set(base.excluded_batches) | set(scenario.excluded_batches)),
            excluded_observations=frozenset(
                set(base.excluded_observations) | set(scenario.excluded_observations)
            ),
            excluded_a_windows={**base.excluded_a_windows, **scenario.excluded_a_windows},
        )
        result = calibrate_spatiotemporal(calibration_input, convention, version, scenario_config)
        rotation_change, translation_change = _transform_changes(full, result)
        offset_change = None
        if result.time_offset.offset_seconds is not None and full.time_offset.offset_seconds is not None:
            offset_change = float(result.time_offset.offset_seconds - full.time_offset.offset_seconds)
        basis = [
            f"full_spatial_status:{full.spatial_status}",
            f"scenario_spatial_status:{result.spatial_status}",
            f"full_time_offset_status:{full.time_offset.status}",
            f"scenario_time_offset_status:{result.time_offset.status}",
            f"excluded_batches:{len(scenario.excluded_batches)}",
            f"excluded_observations:{len(scenario.excluded_observations)}",
            f"excluded_time_windows:{sum(len(v) for v in scenario.excluded_a_windows.values())}",
        ]
        if full.time_offset.status == "identifiable" and result.time_offset.status != "identifiable":
            basis.append("scenario_loses_offset_identifiability")
        if full.success and not result.success:
            basis.append("scenario_loses_joint_identifiability")
        comparisons.append(
            SpatiotemporalComparison(
                label=label,
                mode=result.mode,
                result=result,
                offset_change_seconds=offset_change,
                rotation_change=rotation_change,
                translation_change=translation_change,
                judgment_basis=tuple(basis),
            )
        )
    return full, tuple(comparisons)
