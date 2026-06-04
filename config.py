"""Central configuration for PPO turbulence experiments.

The defaults are intentionally research-grade rather than smoke-test sized:
running ``python run_training.py`` trains one PPO model per turbulence level
for 1,000,000 environment steps. Command-line flags in the runner scripts can
override the expensive values for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class UAVDynamicsConfig:
    """Configuration for the linear 2D UAV double-integrator system."""

    dt: float = 0.1
    episode_length: int = 200
    max_acceleration: float = 5.0
    initial_state_std: float = 0.5
    success_radius: float = 0.5
    disturbance_on_vy_only: bool = True

    @property
    def a_matrix(self) -> np.ndarray:
        """Return the discrete-time state transition matrix."""

        dt = self.dt
        return np.array(
            [
                [1.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @property
    def b_matrix(self) -> np.ndarray:
        """Return the discrete-time control matrix."""

        dt = self.dt
        return np.array(
            [
                [0.0, 0.0],
                [dt, 0.0],
                [0.0, 0.0],
                [0.0, dt],
            ],
            dtype=np.float32,
        )

    @property
    def e_matrix(self) -> np.ndarray:
        """Return the disturbance injection matrix.

        The original paper and reproduction script inject scalar crosswind
        turbulence into the vertical velocity channel, ``vy``.
        """

        if self.disturbance_on_vy_only:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return np.eye(4, dtype=np.float32)

    @property
    def q_matrix(self) -> np.ndarray:
        """Return the quadratic state-cost matrix."""

        return np.eye(4, dtype=np.float32)

    @property
    def r_matrix(self) -> np.ndarray:
        """Return the quadratic action-cost matrix."""

        return 0.1 * np.eye(2, dtype=np.float32)


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for Stable-Baselines3 PPO."""

    total_timesteps: int = 1_000_000
    seed: int = 0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_hidden_sizes: tuple[int, int] = (64, 64)
    value_hidden_sizes: tuple[int, int] = (64, 64)
    activation_fn_name: str = "Tanh"
    device: str = "cpu"


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for model evaluation and GAE-bias measurement."""

    eval_episodes: int = 30
    gae_episodes: int = 200
    deterministic_eval: bool = True
    nominal_policy_sigma: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    sigmas: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00)
    dynamics: UAVDynamicsConfig = field(default_factory=UAVDynamicsConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    models_dir: Path = ROOT_DIR / "models"
    results_dir: Path = ROOT_DIR / "results"
    figures_dir: Path = ROOT_DIR / "figures"
    logs_dir: Path = ROOT_DIR / "results" / "training_logs"

    def ensure_directories(self) -> None:
        """Create all output directories used by the pipeline."""

        for path in (self.models_dir, self.results_dir, self.figures_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


CONFIG = ExperimentConfig()


def sigma_to_name(sigma: float) -> str:
    """Return a filesystem-safe name for a turbulence level."""

    return f"{sigma:.2f}".replace(".", "_")


def model_path(sigma: float, config: ExperimentConfig = CONFIG) -> Path:
    """Return the Stable-Baselines3 model path for a turbulence level."""

    return config.models_dir / f"ppo_uav_sigma_{sigma_to_name(sigma)}.zip"


def training_log_dir(sigma: float, config: ExperimentConfig = CONFIG) -> Path:
    """Return the SB3 logger directory for a turbulence level."""

    return config.logs_dir / f"sigma_{sigma_to_name(sigma)}"


def iter_sigmas(config: ExperimentConfig = CONFIG) -> Iterable[float]:
    """Iterate over configured turbulence levels."""

    return config.sigmas
