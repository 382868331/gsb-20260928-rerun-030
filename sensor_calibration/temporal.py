from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibration import CalibrationConfig, CalibrationVersion, CoordinateConvention, _calibrate_core
from .joint import _batch_fingerprint
from .models import (
    AdoptedTimeInterval,
    CalibrationInput,
    MatchedObservation,
    Observation,
    TemporalCalibrationResult,
    TemporalScenario,
    TemporalScenarioComparison,
)

TIME_UNIT_TO_SECONDS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


@dataclass(frozen=True)
class TemporalCalibrationConfig:
    mode: str = "joint_estimate"
    fixed_offset_seconds: float = 0.0
    simultaneous_tolerance_seconds: float = 0.0
    search_range_seconds: float = 1.0
    search_grid_points: int = 241
    # A clock gap longer than this factor times the median sampling period
    # splits the trajectory into a new segment; interpolation never crosses
    # such a gap and nothing is extrapolated beyond segment endpoints.
    gap_factor: float = 5.0
    identifiability_std_limit_seconds: float = 0.05
    curvature_residual_sigma_factor: float = 1.0
    base: CalibrationConfig = field(default_factory=CalibrationConfig)
    excluded_batches: frozenset[str] = frozenset()
    # Windows are expressed in each batch's raw time unit and apply to the
    # sensor-A clock; they model local time spans removed on purpose.
    excluded_windows: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"fixed_offset", "joint_estimate", "simultaneous_only"}:
            raise ValueError("mode must be fixed_offset, joint_estimate or simultaneous_only")
        object.__setattr__(self, "excluded_batches", frozenset(self.excluded_batches))
        normalized = {
            batch_id: tuple(sorted((float(start), float(end)) for start, end in windows))
            for batch_id, windows in self.excluded_windows.items()
        }
        object.__setattr__(self, "excluded_windows", normalized)


@dataclass(frozen=True)
class _Sample:
    observation_id: str
    time_seconds: float
    position: np.ndarray
    sigma: float


@dataclass(frozen=True)
class _Segment:
    index: int
    samples: tuple[_Sample, ...]

    def times(self) -> np.ndarray:
        return np.array([sample.time_seconds for sample in self.samples], dtype=float)


def _build_segments(
    observations: tuple[Observation, ...],
    unit_factor: float,
    clock: str,
    gap_factor: float,
) -> tuple[tuple[_Segment, ...], tuple[tuple[float, float], ...]]:
    stamped = []
    for observation in observations:
        value = getattr(observation, f"timestamp_{clock}")
        if value is None:
            continue
        sigma = observation.position_sigma if observation.position_sigma is not None else 1.0
        stamped.append(
            _Sample(
                observation_id=observation.observation_id,
                time_seconds=float(value) * unit_factor,
                position=observation.position(clock),
                sigma=float(sigma),
            )
        )
    # Physical results must not depend on the record order; sort by clock.
    stamped.sort(key=lambda sample: (sample.time_seconds, sample.observation_id))
    if len(stamped) < 2:
        return (tuple(_Segment(index, tuple(stamped)) for index, _ in enumerate(stamped)), ()), ()
    deltas = np.diff([sample.time_seconds for sample in stamped])
    median_delta = float(np.median(deltas))
    gap_threshold = gap_factor * median_delta if median_delta > 0 else np.inf
    segments: list[list[_Sample]] = [[stamped[0]]]
    gaps: list[tuple[float, float]] = []
    for delta, sample in zip(deltas, stamped[1:]):
        if delta > gap_threshold:
            previous = segments[-1][-1]
            gaps.append((previous.time_seconds, sample.time_seconds))
            segments.append([sample])
        else:
            segments[-1].append(sample)
    return tuple(_Segment(index, tuple(members)) for index, members in enumerate(segments)), tuple(gaps)


def _anchor_excluded_by_window(
    time_a_seconds: float,
    windows_raw: tuple[tuple[float, float], ...],
    unit_factor: float,
) -> bool:
    return any(
        start * unit_factor <= time_a_seconds <= end * unit_factor for start, end in windows_raw
    )


def _simultaneous_match(
    anchor: _Sample,
    segments_b: tuple[_Segment, ...],
    tolerance_seconds: float,
    target_time_b_seconds: float,
) -> MatchedObservation | None:
    best: tuple[float, _Sample, _Segment] | None = None
    for segment in segments_b:
        for sample in segment.samples:
            difference = abs(sample.time_seconds - target_time_b_seconds)
            if difference <= tolerance_seconds and (best is None or difference < best[0]):
                best = (difference, sample, segment)
    if best is None:
        return None
    _, sample, segment = best
    matched_sigma = float(np.hypot(anchor.sigma, sample.sigma))
    return MatchedObservation(
        batch_id="",
        anchor_observation_id=anchor.observation_id,
        b_observation_id_before=sample.observation_id,
        b_observation_id_after=sample.observation_id,
        time_a_seconds=anchor.time_seconds,
        matched_time_b_seconds=sample.time_seconds,
        interpolation_fraction=0.0,
        segment_index=segment.index,
        sensor_a_position=anchor.position,
        matched_sensor_b_position=sample.position,
        matched_position_sigma=matched_sigma,
        interpolated=False,
    )


def _interpolated_match(
    anchor: _Sample,
    segments_b: tuple[_Segment, ...],
    target_time_b_seconds: float,
) -> MatchedObservation | None:
    for segment in segments_b:
        times = segment.times()
        if target_time_b_seconds < times[0] or target_time_b_seconds > times[-1]:
            continue  # no extrapolation beyond a segment
        position_index = int(np.searchsorted(times, target_time_b_seconds))
        if position_index < len(times) and np.isclose(times[position_index], target_time_b_seconds):
            sample = segment.samples[position_index]
            matched_sigma = float(np.hypot(anchor.sigma, sample.sigma))
            return MatchedObservation(
                batch_id="",
                anchor_observation_id=anchor.observation_id,
                b_observation_id_before=sample.observation_id,
                b_observation_id_after=sample.observation_id,
                time_a_seconds=anchor.time_seconds,
                matched_time_b_seconds=target_time_b_seconds,
                interpolation_fraction=0.0,
                segment_index=segment.index,
                sensor_a_position=anchor.position,
                matched_sensor_b_position=sample.position,
                matched_position_sigma=matched_sigma,
                interpolated=False,
            )
        right = position_index
        left = position_index - 1
        if not (0 <= left < right <= len(times) - 1):
            continue
        before = segment.samples[left]
        after = segment.samples[right]
        span = after.time_seconds - before.time_seconds
        fraction = (target_time_b_seconds - before.time_seconds) / span
        matched_position = (1.0 - fraction) * before.position + fraction * after.position
        # Linear interpolation variance; the anchor's own uncertainty adds.
        interpolated_variance = (1.0 - fraction) ** 2 * before.sigma**2 + fraction**2 * after.sigma**2
        matched_sigma = float(np.sqrt(anchor.sigma**2 + interpolated_variance))
        return MatchedObservation(
            batch_id="",
            anchor_observation_id=anchor.observation_id,
            b_observation_id_before=before.observation_id,
            b_observation_id_after=after.observation_id,
            time_a_seconds=anchor.time_seconds,
            matched_time_b_seconds=target_time_b_seconds,
            interpolation_fraction=float(fraction),
            segment_index=segment.index,
            sensor_a_position=anchor.position,
            matched_sensor_b_position=matched_position,
            matched_position_sigma=matched_sigma,
            interpolated=True,
        )
    return None


@dataclass(frozen=True)
class _BatchData:
    batch_id: str
    unit_factor: float
    time_unit: str
    anchors: tuple[_Sample, ...]
    segments_b: tuple[_Segment, ...]
    gaps_b: tuple[tuple[float, float], ...]
    gaps_a: tuple[tuple[float, float], ...]
    fingerprint: str = ""


def _prepare_batches(
    calibration_input: CalibrationInput,
    config: TemporalCalibrationConfig,
) -> tuple[tuple[_BatchData, ...], dict[tuple[str, str], str]]:
    prepared: list[_BatchData] = []
    excluded: dict[tuple[str, str], str] = {}
    for batch in sorted(calibration_input.batches, key=lambda item: item.batch_id):
        unit_factor = TIME_UNIT_TO_SECONDS[batch.time_unit]
        good_observations = tuple(
            observation for observation in batch.observations
            if not (config.base.exclude_marked_bad and observation.quality == "bad")
        )
        for observation in batch.observations:
            reference = (batch.batch_id, observation.observation_id)
            if config.base.exclude_marked_bad and observation.quality == "bad":
                excluded[reference] = "marked_bad"
            elif observation.timestamp_a is None or observation.timestamp_b is None:
                excluded[reference] = "missing_timestamp"
        windows = config.excluded_windows.get(batch.batch_id, ())
        anchors: list[_Sample] = []
        for observation in good_observations:
            if observation.timestamp_a is None:
                continue
            reference = (batch.batch_id, observation.observation_id)
            time_a_seconds = float(observation.timestamp_a) * unit_factor
            if _anchor_excluded_by_window(time_a_seconds, windows, unit_factor):
                excluded[reference] = "excluded_by_saved_time_window"
                continue
            anchors.append(
                _Sample(
                    observation_id=observation.observation_id,
                    time_seconds=time_a_seconds,
                    position=observation.sensor_a_position,
                    sigma=(
                        float(observation.position_sigma)
                        if observation.position_sigma is not None
                        else 1.0
                    ),
                )
            )
        anchors.sort(key=lambda sample: (sample.time_seconds, sample.observation_id))
        segments_b, gaps_b = _build_segments(
            good_observations, unit_factor, "b", config.gap_factor
        )
        segments_a, gaps_a = _build_segments(
            good_observations, unit_factor, "a", config.gap_factor
        )
        prepared.append(
            _BatchData(
                batch_id=batch.batch_id,
                unit_factor=unit_factor,
                time_unit=batch.time_unit,
                anchors=tuple(anchors),
                segments_b=segments_b,
                gaps_b=gaps_b,
                gaps_a=gaps_a,
                fingerprint=_batch_fingerprint(batch),
            )
        )
    return tuple(prepared), excluded


def _match_at_offset(
    prepared: tuple[_BatchData, ...],
    offset_seconds: float,
    config: TemporalCalibrationConfig,
) -> tuple[list[MatchedObservation], dict[tuple[str, str], str]]:
    matched: list[MatchedObservation] = []
    excluded: dict[tuple[str, str], str] = {}
    simultaneous = config.mode == "simultaneous_only"
    tolerance = (
        config.simultaneous_tolerance_seconds
        if simultaneous
        else 0.0
    )
    for batch in prepared:
        for anchor in batch.anchors:
            reference = (batch.batch_id, anchor.observation_id)
            if simultaneous:
                match = _simultaneous_match(
                    anchor, batch.segments_b, tolerance, anchor.time_seconds
                )
            else:
                target_time_b_seconds = anchor.time_seconds - offset_seconds
                match = _interpolated_match(anchor, batch.segments_b, target_time_b_seconds)
            if match is None:
                excluded[reference] = (
                    "no_simultaneous_partner_within_tolerance"
                    if simultaneous
                    else "target_time_falls_in_a_missing_interval_or_outside_overlap"
                )
                continue
            object.__setattr__(match, "batch_id", batch.batch_id)
            matched.append(match)
    return matched, excluded


def _match_weights(matched: list[MatchedObservation]) -> np.ndarray:
    return _match_weights_prepared(matched, None)


def _match_weights_prepared(matched, prepared):
    by_batch: dict[str, list[int]] = {}
    for index, item in enumerate(matched):
        by_batch.setdefault(item.batch_id, []).append(index)
    weights = np.zeros(len(matched), dtype=float)
    fingerprint_by_batch = (
        {batch.batch_id: batch.fingerprint for batch in prepared}
        if prepared is not None
        else {batch_id: batch_id for batch_id in by_batch}
    )
    group_members: dict[str, list[str]] = {}
    for batch_id in by_batch:
        group_members.setdefault(fingerprint_by_batch[batch_id], []).append(batch_id)
    distinct_acquisitions = max(len(group_members), 1)
    for indices in by_batch.values():
        inverse_variance = np.array(
            [1.0 / matched[index].matched_position_sigma**2 for index in indices]
        )
        within = inverse_variance / inverse_variance.sum()
        for position, index in enumerate(indices):
            batch_id = matched[index].batch_id
            acquisition_share = 1.0 / (
                distinct_acquisitions
                * len(group_members[fingerprint_by_batch[batch_id]])
            )
            weights[index] = acquisition_share * within[position]
    return weights


def _weighted_sse(matched: list[MatchedObservation], weights: np.ndarray) -> float:
    from .calibration import _fit_transform

    points_a = np.vstack([item.sensor_a_position for item in matched])
    points_b = np.vstack([item.matched_sensor_b_position for item in matched])
    rigid, *_ = _fit_transform(points_a, points_b, weights)
    if rigid is None:
        return float("inf")
    residuals = np.linalg.norm(rigid.apply(points_a) - points_b, axis=1)
    normalized = weights * len(matched) / weights.sum()
    return float(np.sum(normalized * residuals**2) / len(matched))


def _classify_motion(batch: _BatchData) -> str:
    points = []
    times = []
    sigmas = []
    for segment in batch.segments_b:
        for sample in segment.samples:
            points.append(sample.position)
            times.append(sample.time_seconds)
            sigmas.append(sample.sigma)
    if len(points) < 3:
        return "insufficient"
    points = np.vstack(points)
    times = np.asarray(times)
    noise = max(float(np.median(sigmas)), np.finfo(float).eps)
    spread = float(np.linalg.norm(points - points.mean(axis=0), axis=1).max())
    if spread < 4.0 * noise:
        return "stationary"
    # Residual of a constant-velocity (linear in time) model.
    design = np.column_stack([times, np.ones_like(times)])
    coefficients, _, _, _ = np.linalg.lstsq(design, points, rcond=None)
    linear_residual = float(
        np.sqrt(np.mean(np.sum((points - design @ coefficients) ** 2, axis=1)))
    )
    if linear_residual < 4.0 * noise:
        return "constant_velocity"
    return "accelerated"


def _profile_offset(
    prepared: tuple[_BatchData, ...],
    config: TemporalCalibrationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(
        -config.search_range_seconds,
        config.search_range_seconds,
        config.search_grid_points,
    )
    values = np.empty_like(grid)
    for index, offset in enumerate(grid):
        matched, _ = _match_at_offset(prepared, float(offset), config)
        values[index] = (
            _weighted_sse(matched, _match_weights_prepared(matched, prepared))
            if matched
            else float("inf")
        )
    return grid, values


def _refine_offset(
    prepared: tuple[_BatchData, ...],
    config: TemporalCalibrationConfig,
    grid: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, bool]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0, float("inf"), False
    best_index = int(np.argmin(np.where(finite, values, np.inf)))
    best_offset = float(grid[best_index])
    best_value = float(values[best_index])
    bounded = best_index in (0, len(grid) - 1)
    if 0 < best_index < len(grid) - 1:
        with np.errstate(invalid="ignore"):
            x0, x1, x2 = grid[best_index - 1], grid[best_index], grid[best_index + 1]
            y0, y1, y2 = values[best_index - 1], values[best_index], values[best_index + 1]
            denom = 2 * (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1))
            numer = x2**2 * (y1 - y0) + x1**2 * (y0 - y2) + x0**2 * (y2 - y1)
            vertex = float(numer / denom) if denom != 0 and np.isfinite(denom) else x1
        if x0 <= vertex <= x2 and np.isfinite(vertex):
            matched, _ = _match_at_offset(prepared, float(vertex), config)
            vertex_value = _weighted_sse(matched, _match_weights_prepared(matched, prepared)) if matched else float("inf")
            if np.isfinite(vertex_value) and vertex_value <= best_value:
                return float(vertex), float(vertex_value), bounded
    return best_offset, best_value, bounded


def _adopted_intervals(
    matched: list[MatchedObservation],
    prepared: tuple[_BatchData, ...],
) -> tuple[AdoptedTimeInterval, ...]:
    prepared_by_id = {batch.batch_id: batch for batch in prepared}
    grouped: dict[tuple[str, int], list[MatchedObservation]] = {}
    for item in matched:
        grouped.setdefault((item.batch_id, item.segment_index), []).append(item)
    intervals: list[AdoptedTimeInterval] = []
    for (batch_id, segment_index), members in sorted(grouped.items()):
        batch = prepared_by_id[batch_id]
        factor = batch.unit_factor
        members.sort(key=lambda item: item.time_a_seconds)
        first = members[0]
        last = members[-1]
        b_times = sorted(item.matched_time_b_seconds for item in members)
        intervals.append(
            AdoptedTimeInterval(
                batch_id=batch_id,
                segment_index=segment_index,
                start_time_a_raw=first.time_a_seconds / factor,
                end_time_a_raw=last.time_a_seconds / factor,
                start_time_b_raw=b_times[0] / factor,
                end_time_b_raw=b_times[-1] / factor,
                matched_observation_count=len(members),
            )
        )
    return tuple(intervals)


def calibrate_temporal(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    config: TemporalCalibrationConfig | None = None,
) -> TemporalCalibrationResult:
    config = config or TemporalCalibrationConfig()
    active_input = CalibrationInput(
        calibration_input.dataset_id,
        tuple(
            sorted(
                (
                    batch
                    for batch in calibration_input.batches
                    if batch.batch_id not in config.excluded_batches
                ),
                key=lambda batch: batch.batch_id,
            )
        ),
    )
    prepared, marked_excluded = _prepare_batches(active_input, config)
    per_batch_motion = {batch.batch_id: _classify_motion(batch) for batch in prepared}
    gaps = {
        batch.batch_id: tuple(
            tuple(span / batch.unit_factor for span in gap)
            for gap in batch.gaps_b
        )
        for batch in prepared
        if batch.gaps_b
    }

    reasons: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    if config.mode == "joint_estimate":
        grid, profile = _profile_offset(prepared, config)
        offset, _, bounded = _refine_offset(prepared, config, grid, profile)
        matched, temporal_excluded = _match_at_offset(prepared, offset, config)
    elif config.mode == "fixed_offset":
        offset = config.fixed_offset_seconds
        bounded = False
        matched, temporal_excluded = _match_at_offset(prepared, offset, config)
    else:
        offset = 0.0
        bounded = False
        matched, temporal_excluded = _match_at_offset(prepared, offset, config)

    excluded_observations = dict(marked_excluded)
    for key, reason in temporal_excluded.items():
        excluded_observations.setdefault(key, reason)

    selected = [
        (
            item.batch_id,
            item.anchor_observation_id,
            item.sensor_a_position,
            item.matched_sensor_b_position,
            item.matched_position_sigma,
        )
        for item in matched
    ]
    used_references = tuple((item.batch_id, item.anchor_observation_id) for item in matched)
    weights = _match_weights_prepared(matched, prepared) if matched else None
    spatial = _calibrate_core(
        selected,
        tuple(sorted(excluded_observations)),
        used_references,
        convention,
        version,
        config.base,
        weights,
    )
    intervals = _adopted_intervals(matched, prepared)

    offset_std: float | None = None
    equivalence_range: tuple[float, float] | None = None
    offset_identifiable = False
    if config.mode != "joint_estimate":
        notes.append(f"offset_not_estimated_in_mode_{config.mode}")
    elif not matched:
        reasons.append("no_overlapping_observations_for_any_trial_offset")
    else:
        noise_floor = float(np.median([item.matched_position_sigma for item in matched]))
        rms_floor = spatial.rms_residual if spatial.rms_residual is not None else noise_floor
        step = (2.0 * config.search_range_seconds) / max(config.search_grid_points - 1, 1)
        probe_step = max(step, 1e-6)
        center_sse = _weighted_sse(matched, weights)
        plus_matched, _ = _match_at_offset(prepared, offset + probe_step, config)
        minus_matched, _ = _match_at_offset(prepared, offset - probe_step, config)
        curvature = None
        if plus_matched and minus_matched and len(plus_matched) == len(minus_matched) == len(matched):
            plus_sse = _weighted_sse(plus_matched, _match_weights_prepared(plus_matched, prepared))
            minus_sse = _weighted_sse(minus_matched, _match_weights_prepared(minus_matched, prepared))
            curvature = (plus_sse - 2.0 * center_sse + minus_sse) / probe_step**2
        motion_labels = set(per_batch_motion.values())
        if bounded:
            reasons.append("time_offset_estimate_is_at_the_search_boundary_widen_the_search_range")
        if curvature is None or not np.isfinite(curvature) or curvature <= 0:
            reasons.append("offset_cost_profile_has_no_unique_minimum")
        else:
            offset_std = float(np.sqrt(2.0 * max(rms_floor, noise_floor) ** 2 / curvature))
            threshold = center_sse + (
                config.curvature_residual_sigma_factor * max(rms_floor, noise_floor)
            ) ** 2
            inside = grid[np.isfinite(profile) & (profile <= threshold)]
            if len(inside):
                equivalence_range = (float(inside.min()), float(inside.max()))
            accelerated = "accelerated" in motion_labels
            if offset_std > config.identifiability_std_limit_seconds or not accelerated:
                if "stationary" in motion_labels and not accelerated:
                    reasons.append("target_is_stationary_time_offset_is_unobservable")
                elif "constant_velocity" in motion_labels and not accelerated:
                    reasons.append(
                        "uniform_straight_line_motion_makes_offset_and_spatial_phase_indistinguishable"
                    )
                elif "insufficient" in motion_labels and not accelerated:
                    reasons.append("records_are_too_short_to_observe_acceleration")
                else:
                    reasons.append("offset_curvature_is_weak_relative_to_measurement_uncertainty")
            else:
                offset_identifiable = True
                notes.append("offset_determined_from_accelerated_motion_with_curvature_well_above_noise")
        if not offset_identifiable:
            warnings.append(
                "small_residuals_do_not_imply_a_unique_calibration_offset_remains_multi_interpretable"
            )

    informative_batches = {
        batch_id: label
        for batch_id, label in per_batch_motion.items()
        if label == "accelerated"
    }
    if len(informative_batches) >= 2:
        notes.append(
            "multiple_batches_carry_offset_information; the shared offset must fit all of them"
        )
    preferred: dict[str, float | None] = {}
    if config.mode == "joint_estimate":
        for batch in prepared:
            batch_grid, batch_profile = _profile_offset((batch,), config)
            finite_batch = np.isfinite(batch_profile)
            if np.any(finite_batch) and per_batch_motion[batch.batch_id] == "accelerated":
                batch_offset, _, _ = _refine_offset((batch,), config, batch_grid, batch_profile)
                preferred[batch.batch_id] = float(batch_offset)
                disagreement = abs(float(batch_offset) - offset)
                tolerance = (
                    4.0 * max(offset_std, config.identifiability_std_limit_seconds)
                    if offset_std is not None
                    else config.identifiability_std_limit_seconds
                )
                if disagreement > tolerance:
                    notes.append(
                        f"batch '{batch.batch_id}' prefers offset {float(batch_offset):.6g}s, "
                        f"which conflicts with the shared offset {offset:.6g}s"
                    )
                    warnings.append(
                        f"batches_disagree_on_fixed_clock_offset:{batch.batch_id}"
                    )
            else:
                preferred[batch.batch_id] = None
    preferred_values = [value for value in preferred.values() if value is not None]
    if len(preferred_values) >= 2 and max(preferred_values) - min(preferred_values) > (
        4.0 * config.identifiability_std_limit_seconds
    ):
        notes.append(
            "accelerated batches prefer incompatible fixed offsets in the range "
            f"{min(preferred_values):.6g}s..{max(preferred_values):.6g}s; no single clock "
            "offset explains every batch"
        )
        warnings.append("batches_disagree_on_fixed_clock_offset")
    fingerprint_groups: dict[str, list[str]] = {}
    for batch in prepared:
        fingerprint_groups.setdefault(batch.fingerprint, []).append(batch.batch_id)
    for members in fingerprint_groups.values():
        if len(members) > 1:
            notes.append(
                "batches " + ", ".join(f"'{name}'" for name in sorted(members))
                + " are identical acquisitions; they share one acquisition weight"
            )
    for batch_id in config.excluded_batches:
        notes.append(f"batch '{batch_id}' excluded from the temporal fit by configuration")
    for batch_id, windows in config.excluded_windows.items():
        notes.append(
            f"batch '{batch_id}': {len(windows)} local time window(s) excluded by configuration"
        )

    return TemporalCalibrationResult(
        mode=config.mode,
        success=bool(spatial.success and (config.mode != "joint_estimate" or offset_identifiable)),
        transform=spatial.transform,
        time_offset_seconds=offset,
        time_offset_std_seconds=offset_std,
        time_offset_equivalence_range_seconds=equivalence_range,
        equivalence_range_is_search_bounded=bounded,
        offset_identifiable=offset_identifiable,
        spatial_result=spatial,
        matched_observations=tuple(matched),
        adopted_intervals=intervals,
        excluded_temporal_observations=excluded_observations,
        gaps=gaps,
        distribution=spatial.distribution,
        rms_residual=spatial.rms_residual,
        measurement_sigma=spatial.measurement_sigma,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        identifiability_notes=tuple(notes),
        per_batch_motion=per_batch_motion,
        per_batch_preferred_offset_seconds=preferred,
    )


def compare_temporal_scenarios(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    scenarios: dict[str, TemporalScenario],
    config: TemporalCalibrationConfig | None = None,
) -> tuple[TemporalCalibrationResult, tuple[TemporalScenarioComparison, ...]]:
    config = config or TemporalCalibrationConfig()
    full = calibrate_temporal(calibration_input, convention, version, config)
    comparisons: list[TemporalScenarioComparison] = []
    for label, scenario in scenarios.items():
        scenario_config = TemporalCalibrationConfig(
            mode=config.mode,
            fixed_offset_seconds=config.fixed_offset_seconds,
            simultaneous_tolerance_seconds=config.simultaneous_tolerance_seconds,
            search_range_seconds=config.search_range_seconds,
            search_grid_points=config.search_grid_points,
            gap_factor=config.gap_factor,
            identifiability_std_limit_seconds=config.identifiability_std_limit_seconds,
            curvature_residual_sigma_factor=config.curvature_residual_sigma_factor,
            base=config.base,
            excluded_batches=frozenset(set(config.excluded_batches) | set(scenario.excluded_batches)),
            excluded_windows={
                batch_id: tuple(config.excluded_windows.get(batch_id, ())) + tuple(windows)
                for batch_id, windows in scenario.excluded_windows.items()
            },
        )
        result = calibrate_temporal(calibration_input, convention, version, scenario_config)
        rotation_change = translation_change = offset_change = None
        if full.transform is not None and result.transform is not None:
            rotation_change = float(
                np.linalg.norm(result.transform.rotation - full.transform.rotation, ord="fro")
            )
            translation_change = float(
                np.linalg.norm(result.transform.translation - full.transform.translation)
            )
        if full.time_offset_seconds is not None and result.time_offset_seconds is not None:
            offset_change = float(result.time_offset_seconds - full.time_offset_seconds)
        basis = [
            f"full_offset_identifiable:{full.offset_identifiable}",
            f"scenario_offset_identifiable:{result.offset_identifiable}",
            f"full_success:{full.success}",
            f"scenario_success:{result.success}",
            f"excluded_batches:{len(scenario.excluded_batches)}",
            f"excluded_windows:{sum(len(w) for w in scenario.excluded_windows.values())}",
            f"full_matched_count:{len(full.matched_observations)}",
            f"scenario_matched_count:{len(result.matched_observations)}",
        ]
        if full.offset_identifiable and not result.offset_identifiable:
            basis.append("scenario_loses_offset_identifiability")
        if result.reasons:
            basis.extend(f"scenario_reason:{reason}" for reason in result.reasons)
        comparisons.append(
            TemporalScenarioComparison(
                label=label,
                scenario=scenario,
                result=result,
                rotation_change=rotation_change,
                translation_change=translation_change,
                offset_change_seconds=offset_change,
                judgment_basis=tuple(basis),
            )
        )
    return full, tuple(comparisons)
