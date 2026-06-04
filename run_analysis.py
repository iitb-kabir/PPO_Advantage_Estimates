"""Run PPO evaluation, GAE-bias analysis, scaling fits, and figure generation."""

from __future__ import annotations

import argparse
import logging

from analysis.compute_gae_bias import compute_bias_table
from analysis.evaluate_models import evaluate_all_models
from analysis.fit_models import fit_scaling_models
from make_plots import generate_all_plots


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigmas", nargs="*", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--gae-episodes", type=int, default=None)
    parser.add_argument("--policy-sigma", type=float, default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-gae", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """Run the complete analysis pipeline."""

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.skip_evaluation:
        evaluate_all_models(sigmas=args.sigmas, episodes=args.eval_episodes)
    if not args.skip_gae:
        compute_bias_table(
            sigmas=args.sigmas,
            n_episodes=args.gae_episodes,
            policy_sigma=args.policy_sigma,
        )
    fit_scaling_models()
    if not args.skip_plots:
        generate_all_plots()


if __name__ == "__main__":
    main()
