"""Manual GAE-bias analysis for trained PPO UAV policies."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from tqdm.auto import tqdm

from config import CONFIG, ExperimentConfig, model_path
from envs import UAVTurbulenceEnv

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Trajectory:
    """Container for one rollout and critic-value sequence."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    values: np.ndarray


def critic_value(model: PPO, observation: np.ndarray) -> float:
    """Evaluate the PPO critic for one observation."""

    obs_tensor = obs_as_tensor(observation.reshape(1, -1), model.device)
    with torch.no_grad():
        value = model.policy.predict_values(obs_tensor)
    return float(value.cpu().numpy().reshape(-1)[0])


def collect_trajectory(
    model: PPO,
    sigma: float,
    initial_state: np.ndarray,
    config: ExperimentConfig = CONFIG,
    deterministic: bool = True,
    seed: int | None = None,
) -> Trajectory:
    """Collect one trajectory under a trained PPO policy."""

    env = UAVTurbulenceEnv(sigma=sigma, dynamics_config=config.dynamics, seed=seed)
    observation, _ = env.reset(options={"initial_state": initial_state})

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    values: list[float] = []

    for _ in range(config.dynamics.episode_length):
        observations.append(observation.copy())
        values.append(critic_value(model, observation))
        action, _ = model.predict(observation, deterministic=deterministic)
        action_array = np.asarray(action, dtype=np.float32)
        next_observation, reward, terminated, truncated, _ = env.step(action_array)
        actions.append(action_array.copy())
        rewards.append(float(reward))
        observation = next_observation
        if terminated or truncated:
            break

    values.append(0.0)
    env.close()
    return Trajectory(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
    )


def compute_td_residuals(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute one-step TD residuals from rewards and critic values."""

    if len(values) != len(rewards) + 1:
        raise ValueError("values must contain one bootstrap value after the final reward")
    return rewards + gamma * values[1:] - values[:-1]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """Compute Generalized Advantage Estimation manually."""

    deltas = compute_td_residuals(rewards=rewards, values=values, gamma=gamma)
    advantages = np.zeros_like(deltas, dtype=np.float64)
    accumulator = 0.0
    for index in range(len(deltas) - 1, -1, -1):
        accumulator = float(deltas[index] + gamma * gae_lambda * accumulator)
        advantages[index] = accumulator
    return advantages


def compare_nominal_disturbed(
    model: PPO,
    sigma: float,
    config: ExperimentConfig = CONFIG,
    n_episodes: int | None = None,
) -> dict[str, float]:
    """Compare nominal and disturbed GAE for one turbulence intensity."""

    rng = np.random.default_rng(config.ppo.seed + int(round(10_000 * sigma)))
    episode_count = n_episodes or config.analysis.gae_episodes
    abs_errors: list[float] = []
    signed_errors: list[float] = []
    td_abs_errors: list[float] = []
    nominal_advantages: list[float] = []
    disturbed_advantages: list[float] = []

    for episode in tqdm(
        range(episode_count),
        desc=f"GAE bias sigma={sigma:.2f}",
        leave=False,
    ):
        initial_state = rng.normal(
            loc=0.0,
            scale=config.dynamics.initial_state_std,
            size=4,
        ).astype(np.float32)
        nominal = collect_trajectory(
            model=model,
            sigma=0.0,
            initial_state=initial_state,
            config=config,
            deterministic=config.analysis.deterministic_eval,
            seed=config.ppo.seed + episode,
        )
        disturbed = collect_trajectory(
            model=model,
            sigma=sigma,
            initial_state=initial_state,
            config=config,
            deterministic=config.analysis.deterministic_eval,
            seed=config.ppo.seed + episode + 100_000,
        )
        nominal_gae = compute_gae(
            nominal.rewards,
            nominal.values,
            config.ppo.gamma,
            config.ppo.gae_lambda,
        )
        disturbed_gae = compute_gae(
            disturbed.rewards,
            disturbed.values,
            config.ppo.gamma,
            config.ppo.gae_lambda,
        )
        nominal_td = compute_td_residuals(nominal.rewards, nominal.values, config.ppo.gamma)
        disturbed_td = compute_td_residuals(disturbed.rewards, disturbed.values, config.ppo.gamma)

        common = min(len(nominal_gae), len(disturbed_gae))
        advantage_error = disturbed_gae[:common] - nominal_gae[:common]
        td_error = disturbed_td[:common] - nominal_td[:common]
        abs_errors.extend(np.abs(advantage_error).tolist())
        signed_errors.extend(advantage_error.tolist())
        td_abs_errors.extend(np.abs(td_error).tolist())
        nominal_advantages.extend(nominal_gae[:common].tolist())
        disturbed_advantages.extend(disturbed_gae[:common].tolist())

    return {
        "sigma": sigma,
        "episodes": float(episode_count),
        "mean_abs_advantage_error": float(np.mean(abs_errors)),
        "std_abs_advantage_error": float(np.std(abs_errors)),
        "mean_signed_advantage_error": float(np.mean(signed_errors)),
        "mean_abs_td_error": float(np.mean(td_abs_errors)),
        "mean_abs_nominal_advantage": float(np.mean(np.abs(nominal_advantages))),
        "mean_abs_disturbed_advantage": float(np.mean(np.abs(disturbed_advantages))),
    }


def compute_bias_table(
    config: ExperimentConfig = CONFIG,
    sigmas: list[float] | None = None,
    n_episodes: int | None = None,
    policy_sigma: float | None = None,
) -> pd.DataFrame:
    """Compute and save the GAE-bias table for the turbulence ladder."""

    config.ensure_directories()
    policy_level = config.analysis.nominal_policy_sigma if policy_sigma is None else policy_sigma
    path = model_path(policy_level, config)
    if not path.exists():
        raise FileNotFoundError(f"Missing PPO model for GAE analysis: {path}")
    model = PPO.load(path, device=config.ppo.device)
    targets = sigmas if sigmas is not None else list(config.sigmas)

    rows = [
        compare_nominal_disturbed(model=model, sigma=sigma, config=config, n_episodes=n_episodes)
        for sigma in targets
    ]
    table = pd.DataFrame(rows)
    table.insert(1, "policy_sigma", policy_level)
    output_path = config.results_dir / "gae_bias.csv"
    table.to_csv(output_path, index=False)
    LOGGER.info("Saved GAE-bias table to %s", output_path)
    return table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigmas", nargs="*", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--policy-sigma", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    compute_bias_table(sigmas=args.sigmas, n_episodes=args.episodes, policy_sigma=args.policy_sigma)


if __name__ == "__main__":
    main()
