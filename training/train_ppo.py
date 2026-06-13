"""Stable-Baselines3 PPO training utilities for UAV turbulence experiments."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from config import CONFIG, ExperimentConfig, model_path, training_log_dir
from envs import UAVTurbulenceEnv

LOGGER = logging.getLogger(__name__)


class TrainingMetricsCallback(BaseCallback):
    """Collect PPO training losses and KL values from the SB3 logger."""

    def __init__(self, sigma: float) -> None:
        """Initialize the metrics collector."""

        super().__init__()
        self.sigma = sigma
        self.rows: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        """Record logger values whenever SB3 has made them available."""

        values = self.model.logger.name_to_value
        if "train/policy_gradient_loss" not in values:
            return True
        row = {
            "sigma": self.sigma,
            "timesteps": float(self.num_timesteps),
            "policy_loss": float(values.get("train/policy_gradient_loss", np.nan)),
            "value_loss": float(values.get("train/value_loss", np.nan)),
            "approx_kl": float(values.get("train/approx_kl", np.nan)),
            "entropy_loss": float(values.get("train/entropy_loss", np.nan)),
            "explained_variance": float(values.get("train/explained_variance", np.nan)),
            "clip_fraction": float(values.get("train/clip_fraction", np.nan)),
        }
        if not self.rows or self.rows[-1] != row:
            self.rows.append(row)
        return True


class StopAtTimestepsCallback(BaseCallback):
    """Stop training once an exact timestep budget has been reached."""

    def __init__(self, max_timesteps: int) -> None:
        """Initialize the exact timestep stopper."""

        super().__init__()
        self.max_timesteps = max_timesteps

    def _on_step(self) -> bool:
        """Return ``False`` when training should stop."""

        return self.num_timesteps < self.max_timesteps


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and Stable-Baselines3."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))


def make_monitored_env(
    sigma: float,
    config: ExperimentConfig,
    seed: int,
    monitor_dir: Path,
) -> Monitor:
    """Create a monitored UAV environment for SB3 training."""

    env = UAVTurbulenceEnv(sigma=sigma, dynamics_config=config.dynamics, seed=seed)
    return Monitor(env, filename=str(monitor_dir / "monitor.csv"))


def create_ppo_model(
    sigma: float,
    config: ExperimentConfig = CONFIG,
    seed: int = 0,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
) -> PPO:
    """Create a PPO model for one turbulence level and seed."""

    active_n_steps = n_steps or config.ppo.n_steps
    active_batch_size = batch_size or config.ppo.batch_size
    active_n_epochs = n_epochs or config.ppo.n_epochs
    if active_n_steps < 2:
        raise ValueError("PPO n_steps must be at least 2")
    if active_batch_size < 2:
        raise ValueError("PPO batch_size must be at least 2")
    if active_batch_size > active_n_steps:
        raise ValueError("PPO batch_size must be less than or equal to n_steps")

    log_dir = training_log_dir(sigma, config, seed=seed)
    log_dir.mkdir(parents=True, exist_ok=True)
    vec_env = make_vec_env(
        lambda: make_monitored_env(sigma, config, seed, log_dir),
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
        n_steps=active_n_steps,
        batch_size=active_batch_size,
        n_epochs=active_n_epochs,
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


def train_single_sigma(
    sigma: float,
    seed: int,
    config: ExperimentConfig = CONFIG,
    total_timesteps: int | None = None,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
) -> Path:
    """Train and save one PPO model for a turbulence level and seed."""

    config.ensure_directories()
    seed_everything(seed)
    active_total_timesteps = total_timesteps or config.ppo.total_timesteps
    active_n_steps = n_steps or config.ppo.n_steps
    log_dir = training_log_dir(sigma, config, seed=seed)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Training PPO for sigma=%.2f with seed=%d, timesteps=%d, n_steps=%d",
        sigma,
        seed,
        active_total_timesteps,
        active_n_steps,
    )
    model = create_ppo_model(
        sigma=sigma,
        config=config,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )
    metrics_callback = TrainingMetricsCallback(sigma=sigma)
    stop_callback = StopAtTimestepsCallback(max_timesteps=active_total_timesteps)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(active_n_steps, 10_000),
        save_path=str(checkpoint_dir),
        name_prefix=f"ppo_uav_sigma_{sigma:.2f}_seed{seed}".replace(".", "_"),
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    model.learn(
        total_timesteps=active_total_timesteps,
        callback=[metrics_callback, checkpoint_callback, stop_callback],
        progress_bar=True,
    )

    output_path = model_path(sigma, config, seed=seed)
    model.save(output_path)
    model.get_env().close()

    metrics_path = (
        config.results_dir / f"training_metrics_sigma_{sigma:.2f}_seed{seed}.csv"
    )
    pd.DataFrame(metrics_callback.rows).to_csv(metrics_path, index=False)
    LOGGER.info("Saved model to %s", output_path)
    LOGGER.info("Saved training metrics to %s", metrics_path)
    return output_path


def train_all_sigmas(
    config: ExperimentConfig = CONFIG,
    sigmas: list[float] | None = None,
    seeds: list[int] | None = None,
    total_timesteps: int | None = None,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
) -> list[Path]:
    """Train one PPO model for each (turbulence level, seed) combination.

    The same seed set is shared across all turbulence levels so the seed is
    decoupled from sigma, enabling mean +/- CI reporting downstream.
    """

    targets = sigmas if sigmas is not None else list(config.sigmas)
    active_seeds = seeds if seeds is not None else list(config.seeds)
    saved_paths: list[Path] = []
    for seed in active_seeds:
        for sigma in targets:
            saved_paths.append(
                train_single_sigma(
                    sigma=sigma,
                    seed=seed,
                    config=config,
                    total_timesteps=total_timesteps,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                )
            )
    return saved_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for direct training-module execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigmas", nargs="*", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    train_all_sigmas(
        sigmas=args.sigmas,
        seeds=args.seeds,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
    )


if __name__ == "__main__":
    main()
