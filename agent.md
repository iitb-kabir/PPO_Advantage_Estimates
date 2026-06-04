# Agent Workspace Log

## Project Overview

This project studies PPO/GAE degradation under aerodynamic turbulence for a
linear 2D UAV control problem. The original repository provided a NumPy/SciPy
analytical Monte Carlo experiment using double-integrator UAV dynamics, an LQR
controller, and an exact Riccati value function to show that GAE bias scales
approximately as `C sigma^2`. The workspace has now been extended into a
Stable-Baselines3 PPO research framework that can train learned policies at
multiple turbulence intensities and test whether PPO performance degradation
matches the same quadratic law.

## Chat History & Context Summary

- User requested a complete PPO-based research framework, not just the existing
  analytical GAE-bias experiment.
- User specified the environment state `[x, vx, y, vy]`, action `[ax, ay]`,
  dynamics `s_{t+1} = A s_t + B a_t + E w_t`, episode length 200, reward
  `-(s^T Q s + a^T R a)`, `Q = I`, and `R = 0.1 I`.
- User requested PPO with Stable-Baselines3, Tanh MLP networks `64 -> 64` for
  policy and value functions, one million training timesteps, seeded runs, and
  the turbulence ladder `[0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]`.
- The existing `run_experiment.py`, `make_plots.py`, `README.md`, and
  `requirements.txt` were inspected. The dynamics were confirmed to match the
  paper: double-integrator matrices with scalar turbulence injected into `vy`
  through `E = [0, 0, 0, 1]`.
- The provided `IJERT_May26_Sahil.pdf` was briefly extracted with `pypdf` to
  confirm the paper's framing and quadratic-bias hypothesis. The PDF was left
  unmodified and remains untracked.
- User later reported a `git push` non-fast-forward rejection followed by
  `git pull origin` and a successful second `git push`.
- User attached a smoke-test run log showing that PPO training completed but
  `--total-timesteps 5000` ran to 6,144 timesteps because Stable-Baselines3
  rounded up to full 2,048-step rollouts. The analysis completed, but the
  log-log plot was skipped because the smoke test used only one nonzero sigma.

## Work Accomplished

- Created `config.py` with dataclass-based experiment, PPO, dynamics, and
  analysis configuration.
- Created `envs/__init__.py` and `envs/uav_env.py` implementing
  `UAVTurbulenceEnv` as a Gymnasium environment.
- Created `training/__init__.py` and `training/train_ppo.py` implementing
  seeded Stable-Baselines3 PPO training, model saving, checkpointing, logging,
  progress bars, and training-metric CSV output.
- Created `run_training.py` as the top-level training CLI.
- Created `analysis/__init__.py`, `analysis/compute_gae_bias.py`,
  `analysis/evaluate_models.py`, and `analysis/fit_models.py`.
- Implemented manual trajectory collection, critic-value evaluation, TD
  residual computation, GAE computation, nominal-vs-disturbed advantage-error
  comparison, PPO evaluation, loss/KL extraction, quadratic bias fitting, and
  performance-drop power-law fitting.
- Created `run_analysis.py` to run evaluation, GAE-bias computation, model
  fitting, and figure generation.
- Replaced `make_plots.py` with dynamic plotting from PPO analysis CSVs. It
  generates advantage-bias, return, bias-return, log-log, residual, and
  training-curve figures as PNG and PDF.
- Updated `requirements.txt` to include PyTorch, Gymnasium, Stable-Baselines3,
  Pandas, and tqdm.
- Updated `README.md` with installation, training, analysis, figure generation,
  repository structure, and original analytical experiment notes.
- Updated `.gitignore` to ignore generated PPO models, result CSVs, training
  logs, and new generated PPO figures.
- Ran an in-memory Python syntax check across the new Python files; it passed.
- Verified imports for modules that do not require the newly added RL
  dependencies.
- Ran a synthetic CSV smoke test through `analysis.fit_models` and
  `make_plots.generate_all_plots`; the fit/plot path succeeded. Training-curve
  plotting was skipped in that synthetic test because no PPO logs existed.
- Cleaned up temporary smoke-test outputs and generated `__pycache__` files.
- Checked Git state after the reported push issue. Local `main` and
  `origin/main` both point to commit `ddf304f`, and the working tree is clean.
- Added an exact timestep stop callback to PPO training so short runs stop at
  the requested timestep budget.
- Added training CLI overrides for `--n-steps`, `--batch-size`, and
  `--n-epochs` to make smoke tests much faster without changing the full
  research defaults.
- Updated README smoke-test commands to use sigma values `0.0 0.1 0.2` and
  lightweight PPO settings so the log-log plot has at least two positive sigma
  points.

## Key Decisions

- Preserved the original analytical experiment in `run_experiment.py`.
- Matched the original UAV dynamics exactly, including disturbance injection
  into the `vy` channel.
- Used `Stable-Baselines3 PPO` with separate models per turbulence level rather
  than one domain-randomized policy, matching the requested experimental design.
- Used a frozen dataclass configuration in `config.py` so defaults are
  reproducible and CLI overrides are still available for quick smoke tests.
- Used a nominal policy at `sigma=0.0` by default for GAE-bias comparison, with
  `--policy-sigma` available for alternative analysis.
- Defined success as final state norm within `success_radius = 0.5`, recorded
  through the environment `info` dictionary.
- Stored final PPO loss/KL summaries in `results/ppo_evaluation.csv` and
  per-sigma training metrics in `results/training_metrics_sigma_*.csv`.

## Next Steps

- Install dependencies with `pip install -r requirements.txt`.
- Run a short PPO smoke test, for example:
  `python run_training.py --sigmas 0.0 0.1 --total-timesteps 5000`.
- Run a short analysis smoke test after those models exist:
  `python run_analysis.py --sigmas 0.0 0.1 --eval-episodes 3 --gae-episodes 5`.
- Run the full research sweep with `python run_training.py`, then
  `python run_analysis.py`.
- Inspect `results/fit_statistics.csv` to determine whether learned PPO
  advantage bias is quadratic and whether PPO performance drop has exponent
  near 2 or differs from the analytical GAE law.
- No Git repair is currently needed for the reported push issue; future
  non-fast-forward rejections should be handled by pulling/rebasing remote
  changes before pushing.
- Re-run the smoke test with:
  `python run_training.py --sigmas 0.0 0.1 0.2 --total-timesteps 5000 --n-steps 512 --n-epochs 2`
  followed by:
  `python run_analysis.py --sigmas 0.0 0.1 0.2 --eval-episodes 3 --gae-episodes 5`.
