"""Train PPO policies for the configured UAV turbulence ladder."""

from __future__ import annotations

import argparse
import logging

from training.train_ppo import train_all_sigmas


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigmas", nargs="*", type=float, default=None)
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Random seeds, shared across all sigma. Default is 0 1 2 3 4.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override default PPO timesteps. Default is 1,000,000.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override PPO rollout length. Useful for fast smoke tests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override PPO minibatch size. Must be <= n_steps.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="Override PPO optimization epochs per rollout.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """Run PPO training."""

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
