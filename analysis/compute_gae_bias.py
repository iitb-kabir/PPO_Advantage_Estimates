"""Manual GAE-bias analysis for trained PPO UAV policies.

Two policy modes are supported:

* ``nominal``  -- a single policy trained at sigma=0 is run under increasing
  turbulence. This is the *theory test*: it isolates how the advantage estimator
  degrades for a fixed (locally quadratic) value function.
* ``matched``  -- the policy trained *at* each sigma is run at that sigma. This is
  the *deployment test*: it measures the bias actually seen during training.

Reporting both keeps Table I internally consistent (the original paper mixed a
nominal-policy bias column with matched-policy return/success columns).

Heavy-tailed divergence at high sigma makes the mean a poor summary, so we also
report the median, IQR, and the fraction of diverged episodes.
"""

from __future__ import annotations

import argparse
import logging
import math
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


def _ci95(values: np.ndarray) -> float:
    """Return the 95% confidence interval half-width for a sample mean."""

    n = len(values)
    if n < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(n))


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
    seed_base: int = 0,
) -> dict[str, float]:
    """Compare nominal (sigma=0) and disturbed GAE for one turbulence intensity.

    Both trajectories use the *same* policy (``model``); only the disturbance
    differs. Returns mean/median/IQR advantage error plus a divergence rate.
    """

    rng = np.random.default_rng(seed_base + int(round(10_000 * sigma)))
    episode_count = n_episodes or config.analysis.gae_episodes
    abs_errors: list[float] = []
    signed_errors: list[float] = []
    td_abs_errors: list[float] = []
    nominal_advantages: list[float] = []
    disturbed_advantages: list[float] = []
    episode_mean_abs: list[float] = []
    diverged_flags: list[bool] = []

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
            seed=seed_base + episode,
        )
        disturbed = collect_trajectory(
            model=model,
            sigma=sigma,
            initial_state=initial_state,
            config=config,
            deterministic=config.analysis.deterministic_eval,
            seed=seed_base + episode + 100_000,
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
        episode_mean_abs.append(float(np.mean(np.abs(advantage_error))))

        final_norm = (
            float(np.linalg.norm(disturbed.observations[-1]))
            if len(disturbed.observations) > 0
            else 0.0
        )
        diverged_flags.append(final_norm > config.analysis.divergence_state_norm)

    abs_array = np.asarray(abs_errors, dtype=float)
    q25, q75 = (np.percentile(abs_array, [25, 75]) if abs_array.size else (0.0, 0.0))
    return {
        "sigma": sigma,
        "episodes": float(episode_count),
        "mean_abs_advantage_error": float(np.mean(abs_errors)) if abs_errors else 0.0,
        "std_abs_advantage_error": float(np.std(abs_errors)) if abs_errors else 0.0,
        "median_abs_advantage_error": float(np.median(abs_array)) if abs_array.size else 0.0,
        "iqr_abs_advantage_error": float(q75 - q25),
        "mean_signed_advantage_error": float(np.mean(signed_errors)) if signed_errors else 0.0,
        "mean_abs_td_error": float(np.mean(td_abs_errors)) if td_abs_errors else 0.0,
        "mean_abs_nominal_advantage": float(np.mean(np.abs(nominal_advantages)))
        if nominal_advantages
        else 0.0,
        "mean_abs_disturbed_advantage": float(np.mean(np.abs(disturbed_advantages)))
        if disturbed_advantages
        else 0.0,
        "frac_diverged": float(np.mean(diverged_flags)) if diverged_flags else 0.0,
    }


def _load_policy_for(
    sigma: float, seed: int, policy_mode: str, policy_sigma: float, config: ExperimentConfig
) -> PPO:
    """Load the policy used to measure bias at one (sigma, seed)."""

    level = policy_sigma if policy_mode == "nominal" else sigma
    path = model_path(level, config, seed=seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing PPO model for GAE analysis: {path}")
    return PPO.load(path, device=config.ppo.device)


def compute_bias_table(
    config: ExperimentConfig = CONFIG,
    sigmas: list[float] | None = None,
    seeds: list[int] | None = None,
    n_episodes: int | None = None,
    policy_sigma: float | None = None,
    policy_mode: str = "nominal",
) -> pd.DataFrame:
    """Compute and save the GAE-bias table for the turbulence ladder.

    Writes a per-seed raw CSV and a seed-aggregated CSV. The aggregated table for
    ``policy_mode='nominal'`` is written to ``gae_bias.csv`` (consumed by the fit
    and plotting code); ``matched`` is written to ``gae_bias_matched.csv``.
    """

    config.ensure_directories()
    policy_level = config.analysis.nominal_policy_sigma if policy_sigma is None else policy_sigma
    targets = sigmas if sigmas is not None else list(config.sigmas)
    active_seeds = seeds if seeds is not None else list(config.seeds)

    raw_rows: list[dict[str, float]] = []
    for seed in active_seeds:
        try:
            if policy_mode == "nominal":
                model = _load_policy_for(0.0, seed, policy_mode, policy_level, config)
        except FileNotFoundError as exc:
            LOGGER.warning("%s -- skipping seed %d", exc, seed)
            continue
        for sigma in targets:
            if policy_mode == "matched":
                try:
                    model = _load_policy_for(sigma, seed, policy_mode, policy_level, config)
                except FileNotFoundError as exc:
                    LOGGER.warning("%s -- skipping", exc)
                    continue
            row = compare_nominal_disturbed(
                model=model,
                sigma=sigma,
                config=config,
                n_episodes=n_episodes,
                seed_base=config.ppo.seed + seed * 1_000_003,
            )
            row["seed"] = float(seed)
            row["policy_mode"] = policy_mode
            row["policy_sigma"] = policy_level if policy_mode == "nominal" else sigma
            raw_rows.append(row)

    raw = pd.DataFrame(raw_rows)
    suffix = "" if policy_mode == "nominal" else f"_{policy_mode}"
    raw_path = config.results_dir / f"gae_bias_raw{suffix}.csv"
    raw.to_csv(raw_path, index=False)
    if raw.empty:
        raise FileNotFoundError("No models found for GAE-bias analysis.")

    aggregated = _aggregate_bias(raw, policy_mode)
    output_name = "gae_bias.csv" if policy_mode == "nominal" else f"gae_bias{suffix}.csv"
    output_path = config.results_dir / output_name
    aggregated.to_csv(output_path, index=False)
    LOGGER.info("Saved GAE-bias table to %s (raw: %s)", output_path, raw_path)
    return aggregated


def _aggregate_bias(raw: pd.DataFrame, policy_mode: str) -> pd.DataFrame:
    """Aggregate per-seed bias rows into mean +/- 95% CI per sigma."""

    metric_cols = [
        "mean_abs_advantage_error",
        "std_abs_advantage_error",
        "median_abs_advantage_error",
        "iqr_abs_advantage_error",
        "mean_signed_advantage_error",
        "mean_abs_td_error",
        "mean_abs_nominal_advantage",
        "mean_abs_disturbed_advantage",
        "frac_diverged",
    ]
    rows: list[dict[str, float]] = []
    for sigma, group in raw.groupby("sigma"):
        row: dict[str, float] = {"sigma": float(sigma), "n_seeds": float(len(group))}
        if "policy_sigma" in group:
            row["policy_sigma"] = float(group["policy_sigma"].iloc[0])
        for col in metric_cols:
            if col in group:
                row[col] = float(group[col].mean())
        bias = group["mean_abs_advantage_error"].to_numpy(dtype=float)
        row["mean_abs_advantage_error_ci95"] = _ci95(bias)
        row["mean_abs_advantage_error_seed_std"] = float(
            np.std(bias, ddof=1) if len(bias) > 1 else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigmas", nargs="*", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--policy-sigma", type=float, default=None)
    parser.add_argument(
        "--policy-mode", choices=["nominal", "matched"], default="nominal"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    compute_bias_table(
        sigmas=args.sigmas,
        seeds=args.seeds,
        n_episodes=args.episodes,
        policy_sigma=args.policy_sigma,
        policy_mode=args.policy_mode,
    )


if __name__ == "__main__":
    main()
