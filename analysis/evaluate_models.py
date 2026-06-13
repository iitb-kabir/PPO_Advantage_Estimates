"""Evaluate trained PPO UAV policies across turbulence intensities and seeds."""

from __future__ import annotations

import argparse
import logging
import math

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from tqdm.auto import tqdm

from config import CONFIG, ExperimentConfig, model_path, training_log_dir
from envs import UAVTurbulenceEnv

LOGGER = logging.getLogger(__name__)


def _ci95(values: np.ndarray) -> float:
    """Return the 95% confidence interval half-width for a sample mean."""

    n = len(values)
    if n < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(n))


def read_latest_training_metrics(
    sigma: float, seed: int, config: ExperimentConfig = CONFIG
) -> dict[str, float]:
    """Read the latest PPO loss metrics written during training."""

    candidates = [
        config.results_dir / f"training_metrics_sigma_{sigma:.2f}_seed{seed}.csv",
        training_log_dir(sigma, config, seed=seed) / "progress.csv",
    ]
    normalized: dict[str, float] = {
        "policy_loss": np.nan,
        "value_loss": np.nan,
        "approx_kl": np.nan,
    }
    for path in candidates:
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        rename = {
            "train/policy_gradient_loss": "policy_loss",
            "train/value_loss": "value_loss",
            "train/approx_kl": "approx_kl",
        }
        frame = frame.rename(columns=rename)
        for key in normalized:
            if key in frame.columns:
                series = frame[key].dropna()
                if not series.empty:
                    normalized[key] = float(series.iloc[-1])
        if not np.isnan(normalized["value_loss"]):
            break
    return normalized


def evaluate_model(
    sigma: float,
    seed: int,
    config: ExperimentConfig = CONFIG,
    episodes: int | None = None,
) -> dict[str, float]:
    """Evaluate one (sigma, seed) model in an environment with the same sigma."""

    path = model_path(sigma, config, seed=seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing model for sigma={sigma:.2f} seed={seed}: {path}")
    model = PPO.load(path, device=config.ppo.device)
    episode_count = episodes or config.analysis.eval_episodes

    returns: list[float] = []
    final_norms: list[float] = []
    successes: list[bool] = []
    lengths: list[int] = []

    for episode in tqdm(
        range(episode_count), desc=f"Evaluate sigma={sigma:.2f} seed={seed}", leave=False
    ):
        env = UAVTurbulenceEnv(
            sigma=sigma,
            dynamics_config=config.dynamics,
            seed=seed * 100_003 + episode + int(round(1000 * sigma)),
        )
        observation, _ = env.reset()
        total_return = 0.0
        final_info: dict[str, float | bool | list[float]] = {}
        for step in range(config.dynamics.episode_length):
            action, _ = model.predict(
                observation,
                deterministic=config.analysis.deterministic_eval,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            final_info = info
            if terminated or truncated:
                lengths.append(step + 1)
                break
        else:
            lengths.append(config.dynamics.episode_length)
        returns.append(total_return)
        final_norms.append(float(final_info.get("state_norm", np.linalg.norm(observation))))
        successes.append(bool(final_info.get("is_success", False)))
        env.close()

    metrics = read_latest_training_metrics(sigma=sigma, seed=seed, config=config)
    return {
        "sigma": sigma,
        "seed": float(seed),
        "episodes": float(episode_count),
        "average_episode_return": float(np.mean(returns)),
        "std_episode_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_final_state_norm": float(np.mean(final_norms)),
        "mean_episode_length": float(np.mean(lengths)),
        **metrics,
    }


def aggregate_over_seeds(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed evaluation rows into mean +/- 95% CI per sigma."""

    rows: list[dict[str, float]] = []
    for sigma, group in raw.groupby("sigma"):
        ret = group["average_episode_return"].to_numpy(dtype=float)
        suc = group["success_rate"].to_numpy(dtype=float)
        rows.append(
            {
                "sigma": float(sigma),
                "n_seeds": float(len(group)),
                "average_episode_return": float(np.mean(ret)),
                "return_ci95": _ci95(ret),
                # Mean within-run episode std (kept for backward compatibility).
                "std_episode_return": float(group["std_episode_return"].mean()),
                "return_seed_std": float(np.std(ret, ddof=1) if len(ret) > 1 else 0.0),
                "success_rate": float(np.mean(suc)),
                "success_rate_ci95": _ci95(suc),
                "mean_final_state_norm": float(group["mean_final_state_norm"].mean()),
                "mean_episode_length": float(group["mean_episode_length"].mean()),
                "policy_loss": float(group["policy_loss"].mean()),
                "value_loss": float(group["value_loss"].mean()),
                "approx_kl": float(group["approx_kl"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)


def evaluate_all_models(
    config: ExperimentConfig = CONFIG,
    sigmas: list[float] | None = None,
    seeds: list[int] | None = None,
    episodes: int | None = None,
) -> pd.DataFrame:
    """Evaluate every trained (sigma, seed) model and save performance tables."""

    config.ensure_directories()
    targets = sigmas if sigmas is not None else list(config.sigmas)
    active_seeds = seeds if seeds is not None else list(config.seeds)

    raw_rows: list[dict[str, float]] = []
    for sigma in targets:
        for seed in active_seeds:
            path = model_path(sigma, config, seed=seed)
            if not path.exists():
                LOGGER.warning("Missing model, skipping: %s", path)
                continue
            raw_rows.append(
                evaluate_model(sigma=sigma, seed=seed, config=config, episodes=episodes)
            )

    raw = pd.DataFrame(raw_rows)
    raw_path = config.results_dir / "ppo_evaluation_raw.csv"
    raw.to_csv(raw_path, index=False)
    if raw.empty:
        raise FileNotFoundError(
            "No trained models found. Train with run_training.py before evaluating."
        )

    aggregated = aggregate_over_seeds(raw)
    output_path = config.results_dir / "ppo_evaluation.csv"
    aggregated.to_csv(output_path, index=False)
    LOGGER.info("Saved PPO evaluation table to %s (raw: %s)", output_path, raw_path)
    return aggregated


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigmas", nargs="*", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    evaluate_all_models(sigmas=args.sigmas, seeds=args.seeds, episodes=args.episodes)


if __name__ == "__main__":
    main()
