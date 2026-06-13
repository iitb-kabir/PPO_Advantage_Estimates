"""Turbulence-curriculum PPO training and evaluation.

This module trains PPO with the gust intensity ramped from 0 up to
``sigma_max`` over the course of training, comparing three schedules:

* ``constant`` -- train at ``sigma_max`` the whole time (the fixed-sigma baseline).
* ``linear``   -- ``sigma(p) = sigma_max * p``.
* ``sqrt``     -- ``sigma(p) = sigma_max * sqrt(p)`` (theory-motivated: because the
  advantage bias scales as ``C * sigma^2``, this makes the bias grow linearly in
  training progress ``p``).

The point of the experiment is to show that the curriculum implied by Theorem 1
reaches higher success/return at high turbulence than fixed-sigma training.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from config import (
    CONFIG,
    ExperimentConfig,
    curriculum_log_dir,
    curriculum_model_path,
)
from envs import UAVTurbulenceEnv
from training.train_ppo import make_monitored_env, seed_everything

LOGGER = logging.getLogger(__name__)


def schedule_value(schedule: str, progress: float, sigma_max: float) -> float:
    """Return the gust intensity for a schedule at training progress ``p``."""

    progress = float(np.clip(progress, 0.0, 1.0))
    if schedule == "constant":
        return sigma_max
    if schedule == "linear":
        return sigma_max * progress
    if schedule == "sqrt":
        return sigma_max * math.sqrt(progress)
    raise ValueError(f"Unknown curriculum schedule: {schedule}")


class CurriculumCallback(BaseCallback):
    """Ramp the environment gust intensity according to a schedule."""

    def __init__(
        self,
        schedule: str,
        sigma_max: float,
        total_timesteps: int,
        update_freq: int,
    ) -> None:
        """Initialize the curriculum callback."""

        super().__init__()
        self.schedule = schedule
        self.sigma_max = sigma_max
        self.total_timesteps = max(1, total_timesteps)
        self.update_freq = max(1, update_freq)
        self._last_update = -1
        self.history: list[dict[str, float]] = []

    def _apply(self) -> None:
        progress = self.num_timesteps / self.total_timesteps
        sigma = schedule_value(self.schedule, progress, self.sigma_max)
        self.training_env.env_method("set_sigma", sigma)
        self.history.append({"timesteps": float(self.num_timesteps), "sigma": sigma})

    def _on_training_start(self) -> None:
        self._apply()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self.update_freq:
            self._last_update = self.num_timesteps
            self._apply()
        return True


def _build_curriculum_model(
    schedule: str,
    seed: int,
    config: ExperimentConfig,
    n_steps: int | None,
    batch_size: int | None,
    n_epochs: int | None,
) -> PPO:
    """Create a PPO model whose environment sigma is driven by a callback."""

    log_dir = curriculum_log_dir(schedule, config, seed=seed)
    log_dir.mkdir(parents=True, exist_ok=True)
    # Start the env at sigma=0; the callback overrides it from the first rollout.
    vec_env = make_vec_env(
        lambda: make_monitored_env(0.0, config, seed, log_dir),
        n_envs=1,
        seed=seed,
    )
    policy_kwargs: dict[str, Any] = {
        "net_arch": {
            "pi": list(config.ppo.policy_hidden_sizes),
            "vf": list(config.ppo.value_hidden_sizes),
        },
        "activation_fn": torch.nn.Tanh,
    }
    model = PPO(
        "MlpPolicy",
        vec_env,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        learning_rate=config.ppo.learning_rate,
        n_steps=n_steps or config.ppo.n_steps,
        batch_size=batch_size or config.ppo.batch_size,
        n_epochs=n_epochs or config.ppo.n_epochs,
        clip_range=config.ppo.clip_range,
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        seed=seed,
        policy_kwargs=policy_kwargs,
        device=config.ppo.device,
        verbose=1,
    )
    model.set_logger(configure(str(log_dir), ["stdout", "csv"]))
    return model


def train_curriculum(
    schedule: str,
    seed: int,
    config: ExperimentConfig = CONFIG,
    sigma_max: float | None = None,
    total_timesteps: int | None = None,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
) -> Path:
    """Train and save one curriculum PPO model."""

    config.ensure_directories()
    seed_everything(seed)
    active_sigma_max = config.curriculum.sigma_max if sigma_max is None else sigma_max
    active_total = total_timesteps or config.curriculum.total_timesteps

    LOGGER.info(
        "Curriculum training: schedule=%s seed=%d sigma_max=%.2f timesteps=%d",
        schedule,
        seed,
        active_sigma_max,
        active_total,
    )
    model = _build_curriculum_model(schedule, seed, config, n_steps, batch_size, n_epochs)
    callback = CurriculumCallback(
        schedule=schedule,
        sigma_max=active_sigma_max,
        total_timesteps=active_total,
        update_freq=config.curriculum.update_freq,
    )
    model.learn(total_timesteps=active_total, callback=callback, progress_bar=True)

    output_path = curriculum_model_path(schedule, config, seed=seed)
    model.save(output_path)
    model.get_env().close()
    LOGGER.info("Saved curriculum model to %s", output_path)
    return output_path


def evaluate_policy_at_sigma(
    model: PPO,
    sigma: float,
    config: ExperimentConfig,
    episodes: int,
    seed_base: int,
) -> dict[str, float]:
    """Evaluate a loaded policy at one turbulence intensity."""

    returns: list[float] = []
    successes: list[bool] = []
    for episode in range(episodes):
        env = UAVTurbulenceEnv(
            sigma=sigma,
            dynamics_config=config.dynamics,
            seed=seed_base + episode,
        )
        observation, _ = env.reset()
        total_return = 0.0
        info: dict[str, Any] = {}
        for _ in range(config.dynamics.episode_length):
            action, _ = model.predict(
                observation, deterministic=config.analysis.deterministic_eval
            )
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            if terminated or truncated:
                break
        returns.append(total_return)
        successes.append(bool(info.get("is_success", False)))
        env.close()
    return {
        "average_episode_return": float(np.mean(returns)),
        "std_episode_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
    }


def evaluate_curriculum(
    config: ExperimentConfig = CONFIG,
    schedules: list[str] | None = None,
    seeds: list[int] | None = None,
    sigma_max: float | None = None,
    episodes: int | None = None,
) -> pd.DataFrame:
    """Evaluate every curriculum schedule at the deployment turbulence level."""

    config.ensure_directories()
    active_schedules = schedules if schedules is not None else list(config.curriculum.schedules)
    active_seeds = seeds if seeds is not None else list(config.seeds)
    active_sigma_max = config.curriculum.sigma_max if sigma_max is None else sigma_max
    active_episodes = episodes or config.curriculum.eval_episodes

    raw_rows: list[dict[str, float]] = []
    for schedule in active_schedules:
        for seed in active_seeds:
            path = curriculum_model_path(schedule, config, seed=seed)
            if not path.exists():
                LOGGER.warning("Missing curriculum model, skipping: %s", path)
                continue
            model = PPO.load(path, device=config.ppo.device)
            metrics = evaluate_policy_at_sigma(
                model=model,
                sigma=active_sigma_max,
                config=config,
                episodes=active_episodes,
                seed_base=10_000 + seed * 1000,
            )
            raw_rows.append({"schedule": schedule, "seed": float(seed), **metrics})

    raw = pd.DataFrame(raw_rows)
    raw_path = config.results_dir / "curriculum_eval_raw.csv"
    raw.to_csv(raw_path, index=False)

    if raw.empty:
        LOGGER.warning("No curriculum models evaluated; nothing to aggregate")
        return raw

    aggregated = _aggregate_curriculum(raw, active_sigma_max)
    agg_path = config.results_dir / "curriculum_eval.csv"
    aggregated.to_csv(agg_path, index=False)
    LOGGER.info("Saved curriculum evaluation to %s", agg_path)
    return aggregated


def _aggregate_curriculum(raw: pd.DataFrame, sigma_max: float) -> pd.DataFrame:
    """Aggregate per-seed curriculum results into mean +/- 95% CI."""

    rows: list[dict[str, float]] = []
    for schedule, group in raw.groupby("schedule"):
        n = len(group)
        ret = group["average_episode_return"].to_numpy(dtype=float)
        suc = group["success_rate"].to_numpy(dtype=float)
        rows.append(
            {
                "schedule": schedule,
                "eval_sigma": sigma_max,
                "n_seeds": float(n),
                "average_episode_return": float(np.mean(ret)),
                "return_ci95": _ci95(ret),
                "success_rate": float(np.mean(suc)),
                "success_rate_ci95": _ci95(suc),
            }
        )
    return pd.DataFrame(rows)


def _ci95(values: np.ndarray) -> float:
    """Return a 95% confidence interval half-width for a sample mean."""

    n = len(values)
    if n < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(n))


def train_all_curricula(
    config: ExperimentConfig = CONFIG,
    schedules: list[str] | None = None,
    seeds: list[int] | None = None,
    sigma_max: float | None = None,
    total_timesteps: int | None = None,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
) -> list[Path]:
    """Train every (schedule, seed) curriculum combination."""

    active_schedules = schedules if schedules is not None else list(config.curriculum.schedules)
    active_seeds = seeds if seeds is not None else list(config.seeds)
    paths: list[Path] = []
    for seed in active_seeds:
        for schedule in active_schedules:
            paths.append(
                train_curriculum(
                    schedule=schedule,
                    seed=seed,
                    config=config,
                    sigma_max=sigma_max,
                    total_timesteps=total_timesteps,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                )
            )
    return paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entry point: train and evaluate the turbulence curriculum."""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if not args.eval_only:
        train_all_curricula(
            schedules=args.schedules,
            seeds=args.seeds,
            sigma_max=args.sigma_max,
            total_timesteps=args.total_timesteps,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
        )
    evaluate_curriculum(
        schedules=args.schedules,
        seeds=args.seeds,
        sigma_max=args.sigma_max,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
