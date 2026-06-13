"""Validate the quadratic-bias bound against the *learned* neural critic.

Theorem 1 is proven for the exact LQR value function ``V*(s) = -s^T P s``. The
paper claims it extends to "any locally quadratic value function", including the
second-order expansion of a smooth neural critic, but never checks this. This
module closes that gap by:

1. Estimating the local Hessian of the trained PPO critic along nominal
   trajectories (via autograd) and comparing its spectral norm to the LQR value
   of ``2||P||_2``.
2. Verifying that in the *stable* regime (small sigma, before policy collapse)
   the measured advantage bias still scales as sigma^2.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from analysis.compute_gae_bias import collect_trajectory
from analysis.fit_models import fit_power_law
from analysis.lqr_reference import lqr_reference
from config import CONFIG, ExperimentConfig, model_path

LOGGER = logging.getLogger(__name__)


def critic_hessian_at(model: PPO, observation: np.ndarray) -> np.ndarray:
    """Return the Hessian of the critic value w.r.t. the state at one point."""

    obs = torch.as_tensor(observation.reshape(-1), dtype=torch.float32, device=model.device)

    def value_fn(x: torch.Tensor) -> torch.Tensor:
        return model.policy.predict_values(x.reshape(1, -1)).reshape(())

    hessian = torch.autograd.functional.hessian(value_fn, obs)
    return hessian.detach().cpu().numpy()


def sample_nominal_states(
    model: PPO, config: ExperimentConfig, n_episodes: int, rng: np.random.Generator
) -> np.ndarray:
    """Collect states visited by the policy under nominal (sigma=0) dynamics."""

    states: list[np.ndarray] = []
    for episode in range(n_episodes):
        initial_state = rng.normal(0.0, config.dynamics.initial_state_std, size=4).astype(
            np.float32
        )
        traj = collect_trajectory(
            model=model,
            sigma=0.0,
            initial_state=initial_state,
            config=config,
            deterministic=config.analysis.deterministic_eval,
            seed=int(rng.integers(0, 1_000_000)),
        )
        states.extend(list(traj.observations))
    return np.asarray(states, dtype=np.float32)


def analyze_seed(
    seed: int,
    config: ExperimentConfig,
    n_episodes: int,
    max_states: int,
) -> dict[str, float] | None:
    """Compute neural-critic Hessian statistics for one sigma=0 seed."""

    path = model_path(0.0, config, seed=seed)
    if not path.exists():
        LOGGER.warning("Missing nominal model, skipping: %s", path)
        return None
    model = PPO.load(path, device=config.ppo.device)
    rng = np.random.default_rng(config.ppo.seed + seed)
    states = sample_nominal_states(model, config, n_episodes, rng)
    if len(states) > max_states:
        idx = rng.choice(len(states), size=max_states, replace=False)
        states = states[idx]

    norms = np.array([float(np.linalg.norm(critic_hessian_at(model, s), ord=2)) for s in states])
    return {
        "seed": float(seed),
        "n_states": float(len(states)),
        "hessian_spec_norm_mean": float(np.mean(norms)),
        "hessian_spec_norm_std": float(np.std(norms)),
        "hessian_spec_norm_median": float(np.median(norms)),
        "hessian_spec_norm_max": float(np.max(norms)),
    }


def validate_critic_bound(
    config: ExperimentConfig = CONFIG,
    seeds: list[int] | None = None,
    n_episodes: int = 5,
    max_states: int = 400,
    stable_sigma_max: float = 0.2,
) -> pd.DataFrame:
    """Run the neural-critic Hessian and stable-regime scaling validation."""

    config.ensure_directories()
    active_seeds = seeds if seeds is not None else list(config.seeds)
    rows = [
        result
        for seed in active_seeds
        if (result := analyze_seed(seed, config, n_episodes, max_states)) is not None
    ]
    raw = pd.DataFrame(rows)
    raw.to_csv(config.results_dir / "critic_hessian_raw.csv", index=False)

    ref = lqr_reference(config)
    summary: dict[str, float] = {
        "lqr_2P_spectral_norm": 2.0 * ref.p_spectral_norm,
    }
    if not raw.empty:
        summary["neural_hessian_spec_norm_mean"] = float(raw["hessian_spec_norm_mean"].mean())
        summary["neural_hessian_spec_norm_median"] = float(raw["hessian_spec_norm_median"].mean())
        summary["neural_hessian_spec_norm_max"] = float(raw["hessian_spec_norm_max"].max())
        summary["ratio_neural_over_lqr"] = (
            summary["neural_hessian_spec_norm_mean"] / summary["lqr_2P_spectral_norm"]
        )

    # Stable-regime sigma^2 check on the learned critic, read from the bias table.
    bias_path = config.results_dir / "gae_bias.csv"
    if bias_path.exists():
        bias = pd.read_csv(bias_path)
        stable = bias[(bias["sigma"] > 0.0) & (bias["sigma"] <= stable_sigma_max)]
        if len(stable) >= 2:
            fit = fit_power_law(
                stable["sigma"].to_numpy(dtype=float),
                stable["mean_abs_advantage_error"].to_numpy(dtype=float),
            )
            summary["stable_regime_exponent"] = fit["exponent"]
            summary["stable_regime_r2_log"] = fit["r2_log"]
            summary["stable_sigma_max"] = stable_sigma_max

    summary_frame = pd.DataFrame([summary])
    summary_frame.to_csv(config.results_dir / "critic_hessian.csv", index=False)
    LOGGER.info("Saved critic-Hessian validation to %s", config.results_dir / "critic_hessian.csv")
    return summary_frame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-states", type=int, default=400)
    parser.add_argument("--stable-sigma-max", type=float, default=0.2)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    validate_critic_bound(
        seeds=args.seeds,
        n_episodes=args.episodes,
        max_states=args.max_states,
        stable_sigma_max=args.stable_sigma_max,
    )


if __name__ == "__main__":
    main()
