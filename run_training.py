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
        "--total-timesteps",
        type=int,
        default=None,
        help="Override default PPO timesteps. Default is 1,000,000.",
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
    train_all_sigmas(sigmas=args.sigmas, total_timesteps=args.total_timesteps)


if __name__ == "__main__":
    main()
