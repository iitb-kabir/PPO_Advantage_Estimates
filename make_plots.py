"""Generate publication-quality plots for PPO turbulence experiments."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CONFIG, ExperimentConfig, training_log_dir

LOGGER = logging.getLogger(__name__)


def configure_style() -> None:
    """Apply a compact publication plotting style."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "0.8",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    """Save a figure as both PNG and PDF."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s.[png,pdf]", output_base)


def load_required_results(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load analysis CSVs needed by the main figures."""

    bias_path = config.results_dir / "gae_bias.csv"
    eval_path = config.results_dir / "ppo_evaluation.csv"
    residual_path = config.results_dir / "model_residuals.csv"
    missing = [path for path in (bias_path, eval_path, residual_path) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing analysis results. Run run_analysis.py first: {joined}")
    return pd.read_csv(bias_path), pd.read_csv(eval_path), pd.read_csv(residual_path)


def plot_advantage_bias_vs_sigma(
    bias: pd.DataFrame,
    residuals: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    """Plot mean absolute advantage error against turbulence intensity."""

    sigma = bias["sigma"].to_numpy(dtype=float)
    error = bias["mean_abs_advantage_error"].to_numpy(dtype=float)
    std = bias.get("std_abs_advantage_error", pd.Series(np.zeros_like(error))).to_numpy(dtype=float)
    c_bias = float(
        np.sum((sigma**2) * error) / np.sum(sigma**4)
        if np.sum(sigma**4) > 0
        else np.nan
    )
    xs = np.linspace(0.0, max(1e-9, float(np.max(sigma))), 300)

    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    ax.errorbar(
        sigma,
        error,
        yerr=std,
        fmt="o",
        ms=4,
        capsize=2,
        color="black",
        ecolor="0.55",
        label="Measured",
    )
    ax.plot(xs, c_bias * xs**2, color="#1f5a8a", lw=1.7, label=rf"Fit $C\sigma^2$")
    r2 = np.nan
    if "bias_quadratic_prediction" in residuals:
        y = residuals["mean_abs_advantage_error"].to_numpy(dtype=float)
        yhat = residuals["bias_quadratic_prediction"].to_numpy(dtype=float)
        denom = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - np.sum((y - yhat) ** 2) / denom if denom > 0 else np.nan
    ax.set_xlabel(r"Turbulence intensity $\sigma$")
    ax.set_ylabel("Mean absolute advantage error")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(True, ls=":", lw=0.4, alpha=0.65)
    if np.isfinite(r2):
        ax.text(0.04, 0.88, rf"$R^2={r2:.3f}$", transform=ax.transAxes)
    save_figure(fig, config.figures_dir / "fig_advantage_bias_vs_sigma")


def plot_return_vs_sigma(evaluation: pd.DataFrame, config: ExperimentConfig) -> None:
    """Plot PPO evaluation return against turbulence intensity."""

    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    ax.errorbar(
        evaluation["sigma"],
        evaluation["average_episode_return"],
        yerr=evaluation["std_episode_return"],
        fmt="o-",
        lw=1.3,
        ms=4,
        capsize=2,
        color="#305f72",
        ecolor="0.55",
    )
    ax.set_xlabel(r"Turbulence intensity $\sigma$")
    ax.set_ylabel("Average episode return")
    ax.grid(True, ls=":", lw=0.4, alpha=0.65)
    save_figure(fig, config.figures_dir / "fig_ppo_return_vs_sigma")


def plot_bias_vs_return(
    bias: pd.DataFrame,
    evaluation: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    """Plot the relationship between GAE bias and PPO return."""

    merged = pd.merge(evaluation, bias, on="sigma", how="inner")
    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    scatter = ax.scatter(
        merged["mean_abs_advantage_error"],
        merged["average_episode_return"],
        c=merged["sigma"],
        cmap="viridis",
        s=34,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Mean absolute advantage error")
    ax.set_ylabel("Average episode return")
    ax.grid(True, ls=":", lw=0.4, alpha=0.65)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"$\sigma$")
    save_figure(fig, config.figures_dir / "fig_advantage_bias_vs_return")


def plot_loglog_bias(bias: pd.DataFrame, config: ExperimentConfig) -> None:
    """Plot advantage bias in log-log coordinates."""

    sigma = bias["sigma"].to_numpy(dtype=float)
    error = bias["mean_abs_advantage_error"].to_numpy(dtype=float)
    mask = (sigma > 0.0) & (error > 0.0)
    if int(np.sum(mask)) < 2:
        LOGGER.warning("Skipping log-log bias plot because too few positive points exist")
        return
    slope, intercept = np.polyfit(np.log(sigma[mask]), np.log(error[mask]), deg=1)
    xs = np.linspace(float(np.min(sigma[mask])), float(np.max(sigma[mask])), 300)

    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    ax.loglog(sigma[mask], error[mask], "o", ms=4, color="black", label="Measured")
    ax.loglog(
        xs,
        np.exp(intercept) * xs**slope,
        "-",
        lw=1.6,
        color="#1f5a8a",
        label=rf"Power fit $p={slope:.2f}$",
    )
    ax.loglog(
        xs,
        error[mask][-1] * (xs / sigma[mask][-1]) ** 2,
        "--",
        lw=1.0,
        color="#b23a48",
        label="Slope 2 reference",
    )
    ax.set_xlabel(r"Turbulence intensity $\sigma$")
    ax.set_ylabel("Mean absolute advantage error")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.65)
    save_figure(fig, config.figures_dir / "fig_loglog_bias")


def plot_residual_analysis(residuals: pd.DataFrame, config: ExperimentConfig) -> None:
    """Plot residuals for the fitted bias and performance models."""

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=False)
    axes[0].axhline(0.0, color="black", lw=0.7)
    axes[0].plot(
        residuals["sigma"],
        residuals["bias_quadratic_residual"],
        "o-",
        ms=4,
        lw=1.1,
        color="#1f5a8a",
    )
    axes[0].set_xlabel(r"$\sigma$")
    axes[0].set_ylabel("Bias residual")
    axes[0].grid(True, ls=":", lw=0.4, alpha=0.65)

    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].plot(
        residuals["sigma"],
        residuals["performance_drop_residual"],
        "s-",
        ms=4,
        lw=1.1,
        color="#b23a48",
    )
    axes[1].set_xlabel(r"$\sigma$")
    axes[1].set_ylabel("Performance-drop residual")
    axes[1].grid(True, ls=":", lw=0.4, alpha=0.65)
    save_figure(fig, config.figures_dir / "fig_residual_analysis")


def load_training_curve(sigma: float, config: ExperimentConfig) -> pd.DataFrame | None:
    """Load one SB3 training curve if it exists."""

    progress_path = training_log_dir(sigma, config) / "progress.csv"
    if progress_path.exists() and progress_path.stat().st_size > 0:
        frame = pd.read_csv(progress_path)
        if not frame.empty:
            frame["sigma"] = sigma
            return frame
    metrics_path = config.results_dir / f"training_metrics_sigma_{sigma:.2f}.csv"
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        frame = pd.read_csv(metrics_path)
        if not frame.empty:
            frame["time/total_timesteps"] = frame.get("timesteps", np.arange(len(frame)))
            frame["sigma"] = sigma
            return frame
    return None


def plot_training_curves(config: ExperimentConfig) -> None:
    """Plot PPO training curves from SB3 logger CSVs."""

    curves = [load_training_curve(sigma, config) for sigma in config.sigmas]
    curves = [curve for curve in curves if curve is not None]
    if not curves:
        LOGGER.warning("Skipping training-curve plot because no training logs exist")
        return

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.8), sharex=False)
    for curve in curves:
        sigma = float(curve["sigma"].iloc[0])
        steps = curve.get("time/total_timesteps", pd.Series(np.arange(len(curve))))
        label = rf"$\sigma={sigma:.2f}$"
        if "rollout/ep_rew_mean" in curve:
            axes[0].plot(steps, curve["rollout/ep_rew_mean"], lw=1.0, label=label)
        if "train/value_loss" in curve:
            axes[1].plot(steps, curve["train/value_loss"], lw=1.0, label=label)
        elif "value_loss" in curve:
            axes[1].plot(steps, curve["value_loss"], lw=1.0, label=label)
        if "train/approx_kl" in curve:
            axes[2].plot(steps, curve["train/approx_kl"], lw=1.0, label=label)
        elif "approx_kl" in curve:
            axes[2].plot(steps, curve["approx_kl"], lw=1.0, label=label)

    axes[0].set_ylabel("Mean rollout return")
    axes[1].set_ylabel("Value loss")
    axes[2].set_ylabel("Approx. KL")
    for ax in axes:
        ax.set_xlabel("Timesteps")
        ax.grid(True, ls=":", lw=0.4, alpha=0.65)
    axes[-1].legend(loc="best", fontsize=6.5)
    save_figure(fig, config.figures_dir / "fig_training_curves")


def generate_all_plots(config: ExperimentConfig = CONFIG) -> None:
    """Generate all requested figures."""

    configure_style()
    bias, evaluation, residuals = load_required_results(config)
    plot_advantage_bias_vs_sigma(bias, residuals, config)
    plot_return_vs_sigma(evaluation, config)
    plot_bias_vs_return(bias, evaluation, config)
    plot_loglog_bias(bias, config)
    plot_residual_analysis(residuals, config)
    plot_training_curves(config)


def main() -> None:
    """CLI entry point."""

    logging.basicConfig(level=logging.INFO)
    generate_all_plots()


if __name__ == "__main__":
    main()
