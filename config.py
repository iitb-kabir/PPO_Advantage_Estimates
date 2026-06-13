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
    # Turbulence model: "white" (i.i.d. Gaussian gust) or "dryden" (first-order
    # colored / Ornstein-Uhlenbeck gust approximating the Dryden lateral spectrum).
    turbulence_type: str = "white"
    # Correlation time (s) for the colored (Dryden) gust. Only used when
    # turbulence_type == "dryden".
    dryden_tau: float = 1.0

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
    # "auto" selects CUDA when available and falls back to CPU otherwise, so the
    # repository runs unchanged on machines without a GPU.
    device: str = "auto"


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for model evaluation and GAE-bias measurement."""

    eval_episodes: int = 30
    gae_episodes: int = 200
    deterministic_eval: bool = True
    nominal_policy_sigma: float = 0.0
    # An episode is flagged "diverged" when its final state norm exceeds this
    # value. Used to report a robust, heavy-tail-aware divergence rate alongside
    # the mean advantage error.
    divergence_state_norm: float = 5.0


@dataclass(frozen=True)
class CurriculumConfig:
    """Configuration for turbulence-curriculum PPO training.

    The schedules ramp the gust intensity from 0 up to ``sigma_max`` over the
    course of training. ``"sqrt"`` is the theory-motivated schedule: because the
    advantage bias scales as ``C * sigma^2`` (Theorem 1), a ``sigma(p) =
    sigma_max * sqrt(p)`` ramp makes the *bias* grow linearly in training
    progress ``p``. ``"linear"`` and ``"constant"`` are baselines.
    """

    sigma_max: float = 0.5
    total_timesteps: int = 1_000_000
    update_freq: int = 2048
    schedules: tuple[str, ...] = ("constant", "linear", "sqrt")
    eval_episodes: int = 100


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    sigmas: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00)
    # Random seeds used for every turbulence level so that the *same* seed set is
    # shared across all sigma. This decouples the seed from sigma and lets us
    # report mean +/- 95% CI instead of single-run point estimates.
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    dynamics: UAVDynamicsConfig = field(default_factory=UAVDynamicsConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
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


def seed_suffix(seed: int | None) -> str:
    """Return the path suffix encoding a training seed (empty if ``None``)."""

    return "" if seed is None else f"_seed{int(seed)}"


def model_path(sigma: float, config: ExperimentConfig = CONFIG, *, seed: int | None = None) -> Path:
    """Return the Stable-Baselines3 model path for a turbulence level/seed."""

    return config.models_dir / f"ppo_uav_sigma_{sigma_to_name(sigma)}{seed_suffix(seed)}.zip"


def training_log_dir(
    sigma: float, config: ExperimentConfig = CONFIG, *, seed: int | None = None
) -> Path:
    """Return the SB3 logger directory for a turbulence level/seed."""

    return config.logs_dir / f"sigma_{sigma_to_name(sigma)}{seed_suffix(seed)}"


def curriculum_model_path(
    schedule: str, config: ExperimentConfig = CONFIG, *, seed: int | None = None
) -> Path:
    """Return the model path for a curriculum-trained policy."""

    return config.models_dir / f"ppo_uav_curriculum_{schedule}{seed_suffix(seed)}.zip"


def curriculum_log_dir(
    schedule: str, config: ExperimentConfig = CONFIG, *, seed: int | None = None
) -> Path:
    """Return the SB3 logger directory for a curriculum-trained policy."""

    return config.logs_dir / f"curriculum_{schedule}{seed_suffix(seed)}"


def iter_sigmas(config: ExperimentConfig = CONFIG) -> Iterable[float]:
    """Iterate over configured turbulence levels."""

    return config.sigmas
