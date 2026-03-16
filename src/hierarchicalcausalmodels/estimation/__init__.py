"""Per-unit estimators and ATE aggregation for confounder-style HCMs."""

from .per_unit import (
    fit_per_unit_estimators,
    aggregate_per_unit_outputs,
    fit_regressors_per_unit,
    estimate_ate_confounder,
    device_kwargs_for_workers,
)

__all__ = [
    "fit_per_unit_estimators",
    "aggregate_per_unit_outputs",
    "fit_regressors_per_unit",
    "estimate_ate_confounder",
    "device_kwargs_for_workers",
]
