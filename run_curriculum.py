"""Train and evaluate the turbulence-curriculum PPO experiment.

Compares the theory-motivated ``sqrt`` schedule against ``linear`` and the
fixed-sigma (``constant``) baseline, all evaluated at the deployment turbulence
level ``sigma_max``.
"""

from __future__ import annotations

import argparse
import logging

from training.curriculum import evaluate_curriculum, train_all_curricula


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
    """Run curriculum training (unless --eval-only) and evaluation."""

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
