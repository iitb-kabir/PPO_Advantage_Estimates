"""Fit scaling laws for advantage bias and PPO performance degradation.

Fixes over the original analysis:

* Quadratic and free power-law fits are both reported with a *log-space* R^2 and
  an AIC, so the two models are compared on the same footing (the original mixed
  a linear-space quadratic R^2 with a log-space power-law R^2).
* The power-law exponent is also fit *per seed*, giving a mean +/- 95% CI on the
  exponent instead of a single point estimate dominated by one data point.
* The empirical coefficient is compared against the closed-form Theorem-1 bound,
  and the critical-threshold sigma*(f) is recomputed and saved.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from analysis.lqr_reference import lqr_reference
from config import CONFIG, ExperimentConfig

LOGGER = logging.getLogger(__name__)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination in the given space."""

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def aic_from_residuals(residuals: np.ndarray, n_params: int) -> float:
    """Gaussian AIC from a residual vector (lower is better)."""

    n = len(residuals)
    rss = float(np.sum(residuals**2))
    if n == 0 or rss <= 0.0:
        return float("nan")
    return float(n * np.log(rss / n) + 2 * n_params)


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """Fit ``y = c x`` through the origin (linear least squares)."""

    denominator = float(np.sum(x * x))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(x * y) / denominator)


def fit_quadratic_logspace(sigmas: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Fit ``values = C * sigma^2`` with the exponent fixed at 2, in log space."""

    mask = (sigmas > 0.0) & (values > 0.0)
    if int(np.sum(mask)) < 1:
        return {"coefficient": float("nan"), "r2_log": float("nan"), "aic_log": float("nan")}
    log_sigma = np.log(sigmas[mask])
    log_y = np.log(values[mask])
    log_c = float(np.mean(log_y - 2.0 * log_sigma))
    pred_log = log_c + 2.0 * log_sigma
    return {
        "coefficient": float(np.exp(log_c)),
        "r2_log": r_squared(log_y, pred_log),
        "aic_log": aic_from_residuals(log_y - pred_log, n_params=1),
    }


def fit_power_law(sigmas: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Fit ``values = C * sigma^p`` in log-log space."""

    mask = (sigmas > 0.0) & (values > 0.0)
    if int(np.sum(mask)) < 2:
        nan = float("nan")
        return {
            "coefficient": nan,
            "exponent": nan,
            "r2_log": nan,
            "r2_linear": nan,
            "aic_log": nan,
        }
    log_sigma = np.log(sigmas[mask])
    log_y = np.log(values[mask])
    slope, intercept = np.polyfit(log_sigma, log_y, deg=1)
    pred_log = intercept + slope * log_sigma
    pred_linear = np.exp(intercept) * sigmas[mask] ** slope
    return {
        "coefficient": float(np.exp(intercept)),
        "exponent": float(slope),
        "r2_log": r_squared(log_y, pred_log),
        "r2_linear": r_squared(values[mask], pred_linear),
        "aic_log": aic_from_residuals(log_y - pred_log, n_params=2),
    }


def per_seed_exponent(raw_bias: pd.DataFrame) -> dict[str, float]:
    """Fit the power-law exponent per seed and return mean +/- 95% CI."""

    if raw_bias is None or raw_bias.empty or "seed" not in raw_bias:
        return {"mean": float("nan"), "ci95": float("nan"), "n_seeds": 0.0}
    exponents: list[float] = []
    for _, group in raw_bias.groupby("seed"):
        fit = fit_power_law(
            group["sigma"].to_numpy(dtype=float),
            group["mean_abs_advantage_error"].to_numpy(dtype=float),
        )
        if np.isfinite(fit["exponent"]):
            exponents.append(fit["exponent"])
    if not exponents:
        return {"mean": float("nan"), "ci95": float("nan"), "n_seeds": 0.0}
    arr = np.asarray(exponents, dtype=float)
    ci = 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return {"mean": float(np.mean(arr)), "ci95": float(ci), "n_seeds": float(len(arr))}


def fit_scaling_models(config: ExperimentConfig = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the hypothesis-test models and save CSV outputs."""

    bias_path = config.results_dir / "gae_bias.csv"
    eval_path = config.results_dir / "ppo_evaluation.csv"
    raw_bias_path = config.results_dir / "gae_bias_raw.csv"
    if not bias_path.exists():
        raise FileNotFoundError(f"Missing GAE-bias results: {bias_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing PPO-evaluation results: {eval_path}")

    bias = pd.read_csv(bias_path)
    evaluation = pd.read_csv(eval_path)
    raw_bias = pd.read_csv(raw_bias_path) if raw_bias_path.exists() else None
    merged = pd.merge(evaluation, bias, on="sigma", how="inner", suffixes=("_eval", "_bias"))
    if merged.empty:
        raise ValueError("No overlapping sigma values between evaluation and GAE-bias tables")

    sigmas = merged["sigma"].to_numpy(dtype=float)
    advantage_bias = merged["mean_abs_advantage_error"].to_numpy(dtype=float)
    sigma_squared = sigmas**2

    # Quadratic through origin (linear space) -- kept for continuity with the paper.
    c_bias = fit_through_origin(sigma_squared, advantage_bias)
    quadratic_bias_pred = c_bias * sigma_squared
    bias_r2_linear = r_squared(advantage_bias, quadratic_bias_pred)

    quad_log = fit_quadratic_logspace(sigmas, advantage_bias)
    power = fit_power_law(sigmas, advantage_bias)
    seed_exp = per_seed_exponent(raw_bias)

    # Performance drop relative to the nominal (sigma=0) controller.
    nominal_candidates = merged.loc[np.isclose(merged["sigma"], 0.0), "average_episode_return"]
    nominal_return = (
        float(nominal_candidates.iloc[0])
        if not nominal_candidates.empty
        else float(merged["average_episode_return"].max())
    )
    performance_drop = np.maximum(
        0.0, nominal_return - merged["average_episode_return"].to_numpy(dtype=float)
    )
    perf_power = fit_power_law(sigmas, performance_drop)

    fit_rows = [
        {
            "model": "advantage_bias_quadratic_origin",
            "coefficient": c_bias,
            "exponent": 2.0,
            "exponent_ci95": np.nan,
            "r_squared_linear": bias_r2_linear,
            "r_squared_log": quad_log["r2_log"],
            "aic_log": quad_log["aic_log"],
            "n_points": float(len(sigmas)),
            "target": "mean_abs_advantage_error",
        },
        {
            "model": "advantage_bias_power_law",
            "coefficient": power["coefficient"],
            "exponent": power["exponent"],
            "exponent_ci95": np.nan,
            "r_squared_linear": power["r2_linear"],
            "r_squared_log": power["r2_log"],
            "aic_log": power["aic_log"],
            "n_points": float(len(sigmas)),
            "target": "mean_abs_advantage_error",
        },
        {
            "model": "advantage_bias_power_law_perseed",
            "coefficient": np.nan,
            "exponent": seed_exp["mean"],
            "exponent_ci95": seed_exp["ci95"],
            "r_squared_linear": np.nan,
            "r_squared_log": np.nan,
            "aic_log": np.nan,
            "n_points": seed_exp["n_seeds"],
            "target": "mean_abs_advantage_error",
        },
        {
            "model": "performance_drop_power_law",
            "coefficient": perf_power["coefficient"],
            "exponent": perf_power["exponent"],
            "exponent_ci95": np.nan,
            "r_squared_linear": perf_power["r2_linear"],
            "r_squared_log": perf_power["r2_log"],
            "aic_log": perf_power["aic_log"],
            "n_points": float(len(sigmas)),
            "target": "nominal_return_minus_return",
        },
    ]
    fits = pd.DataFrame(fit_rows)

    residuals = merged.copy()
    residuals["bias_quadratic_prediction"] = quadratic_bias_pred
    residuals["bias_quadratic_residual"] = advantage_bias - quadratic_bias_pred
    residuals["bias_power_prediction"] = np.where(
        sigmas > 0.0, power["coefficient"] * sigmas ** power["exponent"], 0.0
    )
    residuals["bias_power_residual"] = advantage_bias - residuals["bias_power_prediction"]
    residuals["performance_drop"] = performance_drop
    residuals["performance_drop_prediction"] = np.where(
        sigmas > 0.0, perf_power["coefficient"] * sigmas ** perf_power["exponent"], 0.0
    )
    residuals["performance_drop_residual"] = (
        performance_drop - residuals["performance_drop_prediction"].to_numpy(dtype=float)
    )

    fits_path = config.results_dir / "fit_statistics.csv"
    residuals_path = config.results_dir / "model_residuals.csv"
    fits.to_csv(fits_path, index=False)
    residuals.to_csv(residuals_path, index=False)

    _write_theory_comparison(config, bias, c_bias)

    LOGGER.info("Saved fit statistics to %s", fits_path)
    LOGGER.info("Saved residuals to %s", residuals_path)
    return fits, residuals


def _write_theory_comparison(
    config: ExperimentConfig, bias: pd.DataFrame, c_emp: float
) -> None:
    """Compare the empirical coefficient with the Theorem-1 bound and save sigma*."""

    ref = lqr_reference(config)
    ratio = c_emp / ref.c_theoretical if ref.c_theoretical else float("nan")
    theory = pd.DataFrame(
        [
            {
                "C_empirical": c_emp,
                "C_theoretical": ref.c_theoretical,
                "ratio_emp_over_theo": ratio,
                "P_spectral_norm": ref.p_spectral_norm,
                "closed_loop_rho": ref.closed_loop_rho,
                "kappa": ref.kappa,
                "dare_iterations": float(ref.dare_iterations),
            }
        ]
    )
    theory.to_csv(config.results_dir / "theory_comparison.csv", index=False)

    # Nominal advantage scale: bias is measured relative to the sigma=0 policy, so
    # the relevant scale is the typical |A^GAE| under nominal dynamics.
    nominal = bias.loc[np.isclose(bias["sigma"], 0.0), "mean_abs_nominal_advantage"]
    a_scale = float(nominal.iloc[0]) if not nominal.empty else float(
        bias["mean_abs_nominal_advantage"].replace(0.0, np.nan).dropna().mean()
    )
    rows = []
    for frac in (0.05, 0.10, 0.25, 0.50):
        sigma_star = float(np.sqrt(frac * a_scale / c_emp)) if c_emp > 0 else float("nan")
        rows.append({"f": frac, "advantage_scale": a_scale, "C_empirical": c_emp, "sigma_star": sigma_star})
    pd.DataFrame(rows).to_csv(config.results_dir / "critical_threshold.csv", index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    fit_scaling_models()


if __name__ == "__main__":
    main()
