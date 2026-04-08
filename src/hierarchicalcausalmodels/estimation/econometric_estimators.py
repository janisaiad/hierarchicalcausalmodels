"""Econometric baseline estimators with optional parallel batch fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .parallel import ParallelBackend, parallel_map


ArrayLike = np.ndarray


@dataclass(frozen=True)
class RegressionResult:
    """Result for an OLS-style linear model fit."""

    coefficient_names: tuple[str, ...]
    coefficients: np.ndarray
    std_errors: np.ndarray
    covariance: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    r_squared: float
    n_obs: int
    dof_resid: int
    label: Optional[str] = None

    def coefficient_dict(self) -> dict[str, float]:
        """Return coefficient estimates keyed by name."""
        return {
            name: float(value)
            for name, value in zip(self.coefficient_names, self.coefficients, strict=False)
        }

    def stderr_dict(self) -> dict[str, float]:
        """Return standard errors keyed by coefficient name."""
        return {
            name: float(value)
            for name, value in zip(self.coefficient_names, self.std_errors, strict=False)
        }


@dataclass(frozen=True)
class OLSModelSpec:
    """Specification for one OLS or reduced-form model."""

    outcome: np.ndarray
    regressors: np.ndarray
    regressor_names: Optional[tuple[str, ...]] = None
    controls: Optional[np.ndarray] = None
    control_names: Optional[tuple[str, ...]] = None
    fixed_effects: Optional[np.ndarray] = None
    fixed_effect_name: str = "fe"
    add_intercept: bool = True
    cov_type: str = "hc1"
    label: Optional[str] = None


@dataclass(frozen=True)
class IV2SLSResult:
    """Result for a 2SLS / IV regression."""

    coefficient_names: tuple[str, ...]
    coefficients: np.ndarray
    std_errors: np.ndarray
    covariance: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    r_squared: float
    n_obs: int
    dof_resid: int
    first_stage_results: tuple[RegressionResult, ...]
    reduced_form_result: RegressionResult
    label: Optional[str] = None

    def coefficient_dict(self) -> dict[str, float]:
        """Return coefficient estimates keyed by name."""
        return {
            name: float(value)
            for name, value in zip(self.coefficient_names, self.coefficients, strict=False)
        }

    def stderr_dict(self) -> dict[str, float]:
        """Return standard errors keyed by coefficient name."""
        return {
            name: float(value)
            for name, value in zip(self.coefficient_names, self.std_errors, strict=False)
        }


@dataclass(frozen=True)
class IV2SLSModelSpec:
    """Specification for one IV / 2SLS model."""

    outcome: np.ndarray
    endogenous: np.ndarray
    instruments: np.ndarray
    endogenous_names: Optional[tuple[str, ...]] = None
    instrument_names: Optional[tuple[str, ...]] = None
    controls: Optional[np.ndarray] = None
    control_names: Optional[tuple[str, ...]] = None
    fixed_effects: Optional[np.ndarray] = None
    fixed_effect_name: str = "fe"
    add_intercept: bool = True
    cov_type: str = "hc1"
    label: Optional[str] = None


def _as_1d(y: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be coercible to a 1D array.")
    return arr


def _as_2d(x: Optional[ArrayLike], n_obs: int, name: str) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    if arr.shape[0] != n_obs:
        raise ValueError(
            f"{name} must have {n_obs} rows, got {arr.shape[0]}."
        )
    return arr


def _normalize_names(
    n_cols: int,
    names: Optional[tuple[str, ...]],
    prefix: str,
) -> tuple[str, ...]:
    if names is None:
        return tuple(f"{prefix}_{idx}" for idx in range(n_cols))
    if len(names) != n_cols:
        raise ValueError(
            f"Expected {n_cols} names for {prefix}, got {len(names)}."
        )
    return names


def _encode_fixed_effects(
    fixed_effects: Optional[np.ndarray],
    n_obs: int,
    fixed_effect_name: str,
) -> tuple[Optional[np.ndarray], tuple[str, ...]]:
    if fixed_effects is None:
        return None, ()

    fe = np.asarray(fixed_effects)
    if fe.ndim != 1:
        raise ValueError("fixed_effects must be a 1D array.")
    if len(fe) != n_obs:
        raise ValueError(
            f"fixed_effects must have length {n_obs}, got {len(fe)}."
        )

    levels = np.unique(fe)
    if len(levels) <= 1:
        return None, ()

    dummies = np.column_stack(
        [(fe == level).astype(float) for level in levels[1:]]
    )
    names = tuple(f"{fixed_effect_name}[{level}]" for level in levels[1:])
    return dummies, names


def _safe_pinv(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(matrix, rcond=1e-10)


def _hc1_scale(n_obs: int, n_params: int) -> float:
    if n_obs <= n_params:
        return 1.0
    return float(n_obs / (n_obs - n_params))


def _stack_blocks(
    *blocks: tuple[Optional[np.ndarray], tuple[str, ...]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    matrices: list[np.ndarray] = []
    names: list[str] = []
    for matrix, block_names in blocks:
        if matrix is None:
            continue
        if matrix.size == 0:
            continue
        matrices.append(matrix)
        names.extend(block_names)
    if not matrices:
        raise ValueError("At least one regressor column is required.")
    return np.column_stack(matrices), tuple(names)


def _prepare_design(
    n_obs: int,
    main: np.ndarray,
    main_names: tuple[str, ...],
    controls: Optional[np.ndarray],
    control_names: tuple[str, ...],
    fixed_effects: Optional[np.ndarray],
    fixed_effect_name: str,
    add_intercept: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    intercept_block = (
        np.ones((n_obs, 1), dtype=float),
        ("intercept",),
    ) if add_intercept else (None, ())
    control_block = (controls, control_names) if controls is not None else (None, ())
    fixed_effect_block = _encode_fixed_effects(
        fixed_effects,
        n_obs,
        fixed_effect_name,
    )
    return _stack_blocks(
        intercept_block,
        (main, main_names),
        control_block,
        fixed_effect_block,
    )


def _linear_statistics(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    cov_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    fitted = x @ beta
    residuals = y - fitted
    n_obs = x.shape[0]
    n_params = x.shape[1]
    dof_resid = max(n_obs - np.linalg.matrix_rank(x), 0)
    xtx_inv = _safe_pinv(x.T @ x)

    cov_type_l = cov_type.lower().strip()
    if cov_type_l == "nonrobust":
        sigma2 = float((residuals @ residuals) / max(n_obs - n_params, 1))
        covariance = sigma2 * xtx_inv
    elif cov_type_l == "hc1":
        meat = x.T @ ((residuals[:, None] ** 2) * x)
        covariance = _hc1_scale(n_obs, n_params) * xtx_inv @ meat @ xtx_inv
    else:
        raise ValueError(
            'cov_type must be either "nonrobust" or "hc1", '
            f"got {cov_type!r}."
        )

    std_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    centered = y - float(np.mean(y))
    denom = float(centered @ centered)
    r_squared = 1.0 - float((residuals @ residuals) / denom) if denom > 0 else 0.0
    return fitted, residuals, covariance, r_squared, dof_resid


def fit_ols(
    outcome: np.ndarray,
    regressors: np.ndarray,
    regressor_names: Optional[tuple[str, ...]] = None,
    controls: Optional[np.ndarray] = None,
    control_names: Optional[tuple[str, ...]] = None,
    fixed_effects: Optional[np.ndarray] = None,
    fixed_effect_name: str = "fe",
    add_intercept: bool = True,
    cov_type: str = "hc1",
    label: Optional[str] = None,
) -> RegressionResult:
    """Fit an OLS regression with optional controls and fixed effects."""
    y = _as_1d(outcome, "outcome")
    x_main = _as_2d(regressors, len(y), "regressors")
    if x_main is None:
        raise ValueError("regressors must not be None.")
    x_controls = _as_2d(controls, len(y), "controls")

    main_names = _normalize_names(x_main.shape[1], regressor_names, "x")
    control_names_n = _normalize_names(
        x_controls.shape[1] if x_controls is not None else 0,
        control_names,
        "control",
    ) if x_controls is not None else ()

    x, coefficient_names = _prepare_design(
        len(y),
        x_main,
        main_names,
        x_controls,
        control_names_n,
        fixed_effects,
        fixed_effect_name,
        add_intercept,
    )

    beta = _safe_pinv(x.T @ x) @ x.T @ y
    fitted, residuals, covariance, r_squared, dof_resid = _linear_statistics(
        x,
        y,
        beta,
        cov_type,
    )

    return RegressionResult(
        coefficient_names=coefficient_names,
        coefficients=beta,
        std_errors=np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        covariance=covariance,
        fitted_values=fitted,
        residuals=residuals,
        r_squared=r_squared,
        n_obs=len(y),
        dof_resid=dof_resid,
        label=label,
    )


def fit_reduced_form(
    outcome: np.ndarray,
    instruments: np.ndarray,
    instrument_names: Optional[tuple[str, ...]] = None,
    controls: Optional[np.ndarray] = None,
    control_names: Optional[tuple[str, ...]] = None,
    fixed_effects: Optional[np.ndarray] = None,
    fixed_effect_name: str = "fe",
    add_intercept: bool = True,
    cov_type: str = "hc1",
    label: Optional[str] = None,
) -> RegressionResult:
    """Fit a reduced-form regression of outcome on instruments and controls."""
    return fit_ols(
        outcome=outcome,
        regressors=instruments,
        regressor_names=instrument_names,
        controls=controls,
        control_names=control_names,
        fixed_effects=fixed_effects,
        fixed_effect_name=fixed_effect_name,
        add_intercept=add_intercept,
        cov_type=cov_type,
        label=label,
    )


def fit_2sls(
    outcome: np.ndarray,
    endogenous: np.ndarray,
    instruments: np.ndarray,
    endogenous_names: Optional[tuple[str, ...]] = None,
    instrument_names: Optional[tuple[str, ...]] = None,
    controls: Optional[np.ndarray] = None,
    control_names: Optional[tuple[str, ...]] = None,
    fixed_effects: Optional[np.ndarray] = None,
    fixed_effect_name: str = "fe",
    add_intercept: bool = True,
    cov_type: str = "hc1",
    label: Optional[str] = None,
) -> IV2SLSResult:
    """Fit a 2SLS / IV regression with optional controls and fixed effects."""
    y = _as_1d(outcome, "outcome")
    d = _as_2d(endogenous, len(y), "endogenous")
    z_excl = _as_2d(instruments, len(y), "instruments")
    if d is None or z_excl is None:
        raise ValueError("endogenous and instruments must not be None.")
    w = _as_2d(controls, len(y), "controls")

    d_names = _normalize_names(d.shape[1], endogenous_names, "endog")
    z_names = _normalize_names(z_excl.shape[1], instrument_names, "instrument")
    w_names = _normalize_names(
        w.shape[1] if w is not None else 0,
        control_names,
        "control",
    ) if w is not None else ()

    w_design, w_design_names = _prepare_design(
        len(y),
        np.zeros((len(y), 0), dtype=float),
        (),
        w,
        w_names,
        fixed_effects,
        fixed_effect_name,
        add_intercept,
    )
    x = np.column_stack([d, w_design])
    z = np.column_stack([z_excl, w_design])
    coefficient_names = d_names + w_design_names

    ztz_inv = _safe_pinv(z.T @ z)
    pz = z @ ztz_inv @ z.T
    a = x.T @ pz @ x
    beta = _safe_pinv(a) @ (x.T @ pz @ y)

    fitted = x @ beta
    residuals = y - fitted
    n_obs = len(y)
    n_params = x.shape[1]
    dof_resid = max(n_obs - np.linalg.matrix_rank(x), 0)

    cov_type_l = cov_type.lower().strip()
    if cov_type_l == "nonrobust":
        sigma2 = float((residuals @ residuals) / max(n_obs - n_params, 1))
        covariance = sigma2 * _safe_pinv(a)
    elif cov_type_l == "hc1":
        middle = z.T @ ((residuals[:, None] ** 2) * z)
        meat = x.T @ z @ ztz_inv @ middle @ ztz_inv @ z.T @ x
        covariance = _hc1_scale(n_obs, n_params) * _safe_pinv(a) @ meat @ _safe_pinv(a)
    else:
        raise ValueError(
            'cov_type must be either "nonrobust" or "hc1", '
            f"got {cov_type!r}."
        )

    std_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    centered = y - float(np.mean(y))
    denom = float(centered @ centered)
    r_squared = 1.0 - float((residuals @ residuals) / denom) if denom > 0 else 0.0

    first_stage_results = tuple(
        fit_ols(
            outcome=d[:, idx],
            regressors=z_excl,
            regressor_names=z_names,
            controls=w,
            control_names=w_names,
            fixed_effects=fixed_effects,
            fixed_effect_name=fixed_effect_name,
            add_intercept=add_intercept,
            cov_type=cov_type,
            label=f"{label or 'iv'}:first_stage:{d_names[idx]}",
        )
        for idx in range(d.shape[1])
    )
    reduced_form_result = fit_reduced_form(
        outcome=y,
        instruments=z_excl,
        instrument_names=z_names,
        controls=w,
        control_names=w_names,
        fixed_effects=fixed_effects,
        fixed_effect_name=fixed_effect_name,
        add_intercept=add_intercept,
        cov_type=cov_type,
        label=f"{label or 'iv'}:reduced_form",
    )

    return IV2SLSResult(
        coefficient_names=coefficient_names,
        coefficients=beta,
        std_errors=std_errors,
        covariance=covariance,
        fitted_values=fitted,
        residuals=residuals,
        r_squared=r_squared,
        n_obs=n_obs,
        dof_resid=dof_resid,
        first_stage_results=first_stage_results,
        reduced_form_result=reduced_form_result,
        label=label,
    )


def _fit_ols_spec(spec: OLSModelSpec) -> RegressionResult:
    return fit_ols(
        outcome=spec.outcome,
        regressors=spec.regressors,
        regressor_names=spec.regressor_names,
        controls=spec.controls,
        control_names=spec.control_names,
        fixed_effects=spec.fixed_effects,
        fixed_effect_name=spec.fixed_effect_name,
        add_intercept=spec.add_intercept,
        cov_type=spec.cov_type,
        label=spec.label,
    )


def _fit_iv_spec(spec: IV2SLSModelSpec) -> IV2SLSResult:
    return fit_2sls(
        outcome=spec.outcome,
        endogenous=spec.endogenous,
        instruments=spec.instruments,
        endogenous_names=spec.endogenous_names,
        instrument_names=spec.instrument_names,
        controls=spec.controls,
        control_names=spec.control_names,
        fixed_effects=spec.fixed_effects,
        fixed_effect_name=spec.fixed_effect_name,
        add_intercept=spec.add_intercept,
        cov_type=spec.cov_type,
        label=spec.label,
    )


def fit_many_ols(
    specs: list[OLSModelSpec],
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "threads",
) -> list[RegressionResult]:
    """Fit many OLS specifications in parallel."""
    return parallel_map(
        specs,
        _fit_ols_spec,
        n_jobs=n_jobs,
        backend=parallel_backend,
    )


def fit_many_reduced_form(
    specs: list[OLSModelSpec],
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "threads",
) -> list[RegressionResult]:
    """Alias for fitting many reduced-form specifications."""
    return fit_many_ols(
        specs,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
    )


def fit_many_2sls(
    specs: list[IV2SLSModelSpec],
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "threads",
) -> list[IV2SLSResult]:
    """Fit many 2SLS specifications in parallel."""
    return parallel_map(
        specs,
        _fit_iv_spec,
        n_jobs=n_jobs,
        backend=parallel_backend,
    )
