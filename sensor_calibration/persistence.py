from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .calibration import (
    CalibrationConfig,
    calibrate,
    compare_observation_subsets,
)
from .models import (
    CalibrationInput,
    CalibrationVersion,
    ComparisonPlan,
    CoordinateConvention,
)


def build_plan(
    plan_id: str,
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    config: CalibrationConfig,
    subsets: dict[str, frozenset[tuple[str, str]]] | None = None,
) -> ComparisonPlan:
    return ComparisonPlan(
        plan_id=plan_id,
        dataset_id=calibration_input.dataset_id,
        convention_id=convention.convention_id,
        method=f"{version.method_name}@{version.method_version}",
        excluded_batches=tuple(sorted(config.excluded_batches)),
        excluded_observations=tuple(sorted(config.additional_excluded_observations)),
        subsets={label: tuple(sorted(refs)) for label, refs in (subsets or {}).items()},
        config_parameters={
            "min_points": config.min_points,
            "rank_tolerance": config.rank_tolerance,
            "near_degenerate_ratio": config.near_degenerate_ratio,
            "condition_limit": config.condition_limit,
            "motion_sigma_factor": config.motion_sigma_factor,
            "residual_sigma_factor": config.residual_sigma_factor,
            "influence_score_limit": config.influence_score_limit,
            "influence_relative_factor": config.influence_relative_factor,
            "batch_bias_sigma_factor": config.batch_bias_sigma_factor,
            "batch_transform_sigma_factor": config.batch_transform_sigma_factor,
            "exclude_marked_bad": config.exclude_marked_bad,
        },
    )


def _input_as_dict(calibration_input: CalibrationInput) -> dict[str, Any]:
    return {
        "dataset_id": calibration_input.dataset_id,
        "batches": [
            {
                "batch_id": batch.batch_id,
                "sensor_a_id": batch.sensor_a_id,
                "sensor_b_id": batch.sensor_b_id,
                "length_unit": batch.length_unit,
                "length_scale_to_convention": batch.length_scale_to_convention,
                "acquisition_id": batch.acquisition_id,
                "observations": [
                    {
                        "observation_id": observation.observation_id,
                        "target_id": observation.target_id,
                        "sensor_a_position": observation.sensor_a_position.tolist(),
                        "sensor_b_position": observation.sensor_b_position.tolist(),
                        "timestamp": observation.timestamp,
                        "quality": observation.quality,
                        "position_sigma": observation.position_sigma,
                    }
                    for observation in batch.observations
                ],
            }
            for batch in calibration_input.batches
        ],
    }


def save_comparison_bundle(
    path: str | Path,
    plan: ComparisonPlan,
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    config: CalibrationConfig | None = None,
) -> dict[str, Any]:
    """Run the plan and persist it together with full traceability data.

    The saved bundle keeps every original observation (nothing is deleted by
    an exclusion), the adoption decision per point, the joint result and
    every named subset comparison.  A reader can trace each point back to its
    acquisition and see exactly which decision was applied.
    """
    base_config = config or CalibrationConfig()
    full_result, comparisons = compare_observation_subsets(
        calibration_input,
        convention,
        version,
        {label: set(refs) for label, refs in plan.subsets.items()},
        CalibrationConfig(
            min_points=base_config.min_points,
            rank_tolerance=base_config.rank_tolerance,
            near_degenerate_ratio=base_config.near_degenerate_ratio,
            condition_limit=base_config.condition_limit,
            motion_sigma_factor=base_config.motion_sigma_factor,
            residual_sigma_factor=base_config.residual_sigma_factor,
            influence_score_limit=base_config.influence_score_limit,
            influence_relative_factor=base_config.influence_relative_factor,
            exclude_marked_bad=base_config.exclude_marked_bad,
            compute_influence=base_config.compute_influence,
            excluded_batches=frozenset(plan.excluded_batches),
            additional_excluded_observations=frozenset(plan.excluded_observations),
            batch_bias_sigma_factor=base_config.batch_bias_sigma_factor,
            batch_transform_sigma_factor=base_config.batch_transform_sigma_factor,
            evaluate_batches=base_config.evaluate_batches,
        ),
    )
    bundle = {
        "schema": "sensor-calibration-comparison-bundle/1.0",
        "plan": plan.as_dict(),
        "input": _input_as_dict(calibration_input),
        "convention": {
            "convention_id": convention.convention_id,
            "frame_a": convention.frame_a,
            "frame_b": convention.frame_b,
            "length_unit": convention.length_unit,
            "handedness": convention.handedness,
            "description": convention.description,
        },
        "version": {
            "method_name": version.method_name,
            "method_version": version.method_version,
            "software_version": version.software_version,
            "parameters": dict(version.parameters),
        },
        "joint_result": full_result.as_dict(),
        "subset_comparisons": [
            {
                "label": item.label,
                "additional_excluded_observations": [
                    list(ref) for ref in item.additional_excluded_observations
                ],
                "rotation_change": item.rotation_change,
                "translation_change": item.translation_change,
                "judgment_basis": list(item.judgment_basis),
                "result": item.result.as_dict(),
            }
            for item in comparisons
        ],
    }
    output_path = Path(path)
    output_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle


def load_comparison_bundle(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
