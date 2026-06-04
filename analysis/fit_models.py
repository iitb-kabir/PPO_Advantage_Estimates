"""Fit scaling laws for advantage bias and PPO performance degradation."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from config import CONFIG, ExperimentConfig

LOGGER = logging.getLogger(__name__)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination."""

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """Fit ``y = c x`` through the origin."""

    denominator = float(np.sum(x * x))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(x * y) / denominator)


def fit_power_law(sigmas: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """Fit ``values = coefficient * sigma ** exponent`` in log-log space."""

    mask = (sigmas > 0.0) & (values > 0.0)
    if int(np.sum(mask)) < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log(sigmas[mask]), np.log(values[mask]), deg=1)
    predictions = np.exp(intercept) * sigmas[mask] ** slope
    return float(np.exp(intercept)), float(slope), r_squared(values[mask], predictions)


def fit_scaling_models(config: ExperimentConfig = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the requested hypothesis-test models and save CSV outputs."""

    bias_path = config.results_dir / "gae_bias.csv"
    eval_path = config.results_dir / "ppo_evaluation.csv"
    if not bias_path.exists():
        raise FileNotFoundError(f"Missing GAE-bias results: {bias_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing PPO-evaluation results: {eval_path}")

    bias = pd.read_csv(bias_path)
    evaluation = pd.read_csv(eval_path)
    merged = pd.merge(evaluation, bias, on="sigma", how="inner")
    if merged.empty:
        raise ValueError("No overlapping sigma values between evaluation and GAE-bias tables")

    sigmas = merged["sigma"].to_numpy(dtype=float)
    advantage_bias = merged["mean_abs_advantage_error"].to_numpy(dtype=float)
    sigma_squared = sigmas**2

    c_bias = fit_through_origin(sigma_squared, advantage_bias)
    quadratic_bias_pred = c_bias * sigma_squared
    bias_r2 = r_squared(advantage_bias, quadratic_bias_pred)
    bias_power_c, bias_power_p, bias_power_r2 = fit_power_law(sigmas, advantage_bias)

    nominal_candidates = merged.loc[np.isclose(merged["sigma"], 0.0), "average_episode_return"]
    nominal_return = (
        float(nominal_candidates.iloc[0])
        if not nominal_candidates.empty
        else float(merged["average_episode_return"].max())
    )
    performance_drop = np.maximum(
        0.0,
        nominal_return - merged["average_episode_return"].to_numpy(dtype=float),
    )
    perf_k, perf_p, perf_r2 = fit_power_law(sigmas, performance_drop)

    fit_rows = [
        {
            "model": "advantage_bias_quadratic_origin",
            "coefficient": c_bias,
            "exponent": 2.0,
            "r_squared": bias_r2,
            "target": "mean_abs_advantage_error",
        },
        {
            "model": "advantage_bias_power_law",
            "coefficient": bias_power_c,
            "exponent": bias_power_p,
            "r_squared": bias_power_r2,
            "target": "mean_abs_advantage_error",
        },
        {
            "model": "performance_drop_power_law",
            "coefficient": perf_k,
            "exponent": perf_p,
            "r_squared": perf_r2,
            "target": "nominal_return_minus_return",
        },
    ]
    fits = pd.DataFrame(fit_rows)

    residuals = merged.copy()
    residuals["bias_quadratic_prediction"] = quadratic_bias_pred
    residuals["bias_quadratic_residual"] = advantage_bias - quadratic_bias_pred
    residuals["performance_drop"] = performance_drop
    residuals["performance_drop_prediction"] = perf_k * np.where(sigmas > 0.0, sigmas**perf_p, 0.0)
    residuals["performance_drop_residual"] = (
        performance_drop - residuals["performance_drop_prediction"].to_numpy(dtype=float)
    )

    fits_path = config.results_dir / "fit_statistics.csv"
    residuals_path = config.results_dir / "model_residuals.csv"
    fits.to_csv(fits_path, index=False)
    residuals.to_csv(residuals_path, index=False)
    LOGGER.info("Saved fit statistics to %s", fits_path)
    LOGGER.info("Saved residuals to %s", residuals_path)
    return fits, residuals


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
