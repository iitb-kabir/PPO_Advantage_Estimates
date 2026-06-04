"""Evaluate trained PPO UAV policies across turbulence intensities."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from tqdm.auto import tqdm

from config import CONFIG, ExperimentConfig, model_path, training_log_dir
from envs import UAVTurbulenceEnv

LOGGER = logging.getLogger(__name__)


def read_latest_training_metrics(sigma: float, config: ExperimentConfig = CONFIG) -> dict[str, float]:
    """Read the latest PPO loss metrics written during training."""

    candidates = [
        config.results_dir / f"training_metrics_sigma_{sigma:.2f}.csv",
        training_log_dir(sigma, config) / "progress.csv",
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
    config: ExperimentConfig = CONFIG,
    episodes: int | None = None,
) -> dict[str, float]:
    """Evaluate the model trained at ``sigma`` in an environment with the same sigma."""

    path = model_path(sigma, config)
    if not path.exists():
        raise FileNotFoundError(f"Missing model for sigma={sigma:.2f}: {path}")
    model = PPO.load(path, device=config.ppo.device)
    episode_count = episodes or config.analysis.eval_episodes

    returns: list[float] = []
    final_norms: list[float] = []
    successes: list[bool] = []
    lengths: list[int] = []

    for episode in tqdm(range(episode_count), desc=f"Evaluate sigma={sigma:.2f}", leave=False):
        env = UAVTurbulenceEnv(
            sigma=sigma,
            dynamics_config=config.dynamics,
            seed=config.ppo.seed + episode + int(round(1000 * sigma)),
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

    metrics = read_latest_training_metrics(sigma=sigma, config=config)
    return {
        "sigma": sigma,
        "episodes": float(episode_count),
        "average_episode_return": float(np.mean(returns)),
        "std_episode_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_final_state_norm": float(np.mean(final_norms)),
        "mean_episode_length": float(np.mean(lengths)),
        **metrics,
    }


def evaluate_all_models(
    config: ExperimentConfig = CONFIG,
    sigmas: list[float] | None = None,
    episodes: int | None = None,
) -> pd.DataFrame:
    """Evaluate every trained model and save the performance table."""

    config.ensure_directories()
    targets = sigmas if sigmas is not None else list(config.sigmas)
    rows = [evaluate_model(sigma=sigma, config=config, episodes=episodes) for sigma in targets]
    table = pd.DataFrame(rows)
    output_path = config.results_dir / "ppo_evaluation.csv"
    table.to_csv(output_path, index=False)
    LOGGER.info("Saved PPO evaluation table to %s", output_path)
    return table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigmas", nargs="*", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    evaluate_all_models(sigmas=args.sigmas, episodes=args.episodes)


if __name__ == "__main__":
    main()
