"""Gymnasium environment for a linear UAV under Gaussian turbulence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import UAVDynamicsConfig


@dataclass(frozen=True)
class StepDiagnostics:
    """Diagnostic values returned in the environment ``info`` dictionary."""

    state_cost: float
    action_cost: float
    disturbance: float | list[float]
    state_norm: float
    is_success: bool

    def as_dict(self) -> dict[str, float | bool | list[float]]:
        """Convert diagnostics to a plain dictionary."""

        return {
            "state_cost": self.state_cost,
            "action_cost": self.action_cost,
            "disturbance": self.disturbance,
            "state_norm": self.state_norm,
            "is_success": self.is_success,
        }


class UAVTurbulenceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Linear 2D UAV control task with configurable turbulence.

    State:
        ``[x, vx, y, vy]``

    Action:
        ``[ax, ay]``, clipped to ``[-max_acceleration, max_acceleration]``.

    Dynamics:
        ``s_{t+1} = A s_t + B a_t + E w_t``, where ``w_t`` is Gaussian with
        standard deviation ``sigma``. By default, ``E`` matches the original
        repository and injects a scalar disturbance into ``vy``.

    Reward:
        ``-(s^T Q s + a^T R a)`` with ``Q = I`` and ``R = 0.1 I``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sigma: float = 0.0,
        dynamics_config: UAVDynamicsConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            sigma: Standard deviation of Gaussian turbulence.
            dynamics_config: Linear-system and reward configuration.
            seed: Optional random seed for reproducible resets and noise.
        """

        super().__init__()
        self.sigma = float(sigma)
        self.config = dynamics_config or UAVDynamicsConfig()
        self.A = self.config.a_matrix
        self.B = self.config.b_matrix
        self.E = self.config.e_matrix
        self.Q = self.config.q_matrix
        self.R = self.config.r_matrix
        self.current_step = 0
        self.state = np.zeros(4, dtype=np.float32)

        state_bound = np.full(4, np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-state_bound, state_bound, dtype=np.float32)
        action_bound = np.full(2, self.config.max_acceleration, dtype=np.float32)
        self.action_space = spaces.Box(-action_bound, action_bound, dtype=np.float32)
        self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the episode to a random initial condition."""

        super().reset(seed=seed)
        if options is not None and "initial_state" in options:
            initial_state = np.asarray(options["initial_state"], dtype=np.float32)
            if initial_state.shape != (4,):
                raise ValueError("initial_state must have shape (4,)")
            self.state = initial_state.copy()
        else:
            self.state = self.np_random.normal(
                loc=0.0,
                scale=self.config.initial_state_std,
                size=4,
            ).astype(np.float32)
        self.current_step = 0
        return self.state.copy(), {"sigma": self.sigma}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the UAV dynamics by one step."""

        clipped_action = np.clip(action, self.action_space.low, self.action_space.high).astype(
            np.float32
        )
        reward = self.reward(self.state, clipped_action)
        disturbance_sample = self._sample_disturbance()
        next_state = self.A @ self.state + self.B @ clipped_action + disturbance_sample

        self.state = next_state.astype(np.float32)
        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.config.episode_length
        diagnostics = self._diagnostics(clipped_action, disturbance_sample)
        return self.state.copy(), reward, terminated, truncated, diagnostics.as_dict()

    def reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Compute the quadratic control reward for a state-action pair."""

        state_cost = float(state.T @ self.Q @ state)
        action_cost = float(action.T @ self.R @ action)
        return -(state_cost + action_cost)

    def state_cost(self, state: np.ndarray) -> float:
        """Return ``s^T Q s`` for diagnostics and tests."""

        return float(state.T @ self.Q @ state)

    def action_cost(self, action: np.ndarray) -> float:
        """Return ``a^T R a`` for diagnostics and tests."""

        return float(action.T @ self.R @ action)

    def _sample_disturbance(self) -> np.ndarray:
        """Sample and shape Gaussian turbulence for the linear dynamics."""

        if self.sigma == 0.0:
            return np.zeros(4, dtype=np.float32)
        if self.E.ndim == 1:
            w = float(self.np_random.normal(loc=0.0, scale=self.sigma))
            return (self.E * w).astype(np.float32)
        w_vec = self.np_random.normal(loc=0.0, scale=self.sigma, size=self.E.shape[1])
        return (self.E @ w_vec).astype(np.float32)

    def _diagnostics(self, action: np.ndarray, disturbance: np.ndarray) -> StepDiagnostics:
        """Build diagnostic information for the current transition."""

        state_norm = float(np.linalg.norm(self.state))
        if self.E.ndim == 1:
            disturbance_info: float | list[float] = float(disturbance[3])
        else:
            disturbance_info = disturbance.astype(float).tolist()
        return StepDiagnostics(
            state_cost=self.state_cost(self.state),
            action_cost=self.action_cost(action),
            disturbance=disturbance_info,
            state_norm=state_norm,
            is_success=state_norm <= self.config.success_radius,
        )
