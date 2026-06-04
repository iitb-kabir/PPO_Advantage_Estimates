# PPO Advantage Bias Under UAV Turbulence

This repository studies whether PPO performance degradation under UAV
turbulence follows the same quadratic scaling law previously observed for
Generalized Advantage Estimation (GAE) bias.

The original NumPy/SciPy experiment is still available in `run_experiment.py`.
It uses a linear UAV model, an LQR controller, the exact Riccati value function,
and Monte Carlo simulations to show that advantage-estimation error scales as
`C sigma^2`. The repository now also contains a complete Stable-Baselines3 PPO
training and analysis pipeline for testing the same hypothesis with learned
policies.

## Research Question

Does PPO performance degradation under turbulence follow the same quadratic law
as GAE bias?

The framework trains independent PPO policies for:

```text
sigma = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]
```

For each turbulence level it measures:

- Average episode return
- Success rate
- Policy loss
- Value loss
- Approximate KL divergence
- Manual GAE advantage bias

It then fits:

- `Bias = C sigma^2`
- `Performance Drop = K sigma^p`

and reports `C`, `R^2`, `K`, and the performance exponent `p`.

## Environment

The Gymnasium environment is `envs/uav_env.py::UAVTurbulenceEnv`.

State:

```text
[x, vx, y, vy]
```

Action:

```text
[ax, ay]
```

Dynamics:

```text
s_{t+1} = A s_t + B a_t + E w_t
w_t ~ N(0, sigma^2)
```

The matrices match the original double-integrator experiment:

```text
A = [[1, dt, 0,  0 ],
     [0, 1,  0,  0 ],
     [0, 0,  1,  dt],
     [0, 0,  0,  1 ]]

B = [[0,  0 ],
     [dt, 0 ],
     [0,  0 ],
     [0,  dt]]

E = [0, 0, 0, 1]
```

Reward:

```text
r = -(s^T Q s + a^T R a)
Q = I
R = 0.1 I
```

Episode length is 200 steps.

## Repository Structure

```text
.
|-- envs/
|   |-- __init__.py
|   `-- uav_env.py
|-- training/
|   |-- __init__.py
|   `-- train_ppo.py
|-- analysis/
|   |-- __init__.py
|   |-- compute_gae_bias.py
|   |-- evaluate_models.py
|   `-- fit_models.py
|-- figures/
|-- models/
|-- results/
|-- config.py
|-- run_training.py
|-- run_analysis.py
|-- run_experiment.py
|-- make_plots.py
|-- README.md
`-- requirements.txt
```

## Installation

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

The implementation is CPU-compatible. Stable-Baselines3 will use `device="cpu"`
by default.

## Train PPO Models

Run the full research training sweep:

```bash
python run_training.py
```

Default PPO settings are in `config.py`:

- Timesteps per sigma: `1,000,000`
- Policy network: `64 -> 64`
- Value network: `64 -> 64`
- Activation: `Tanh`
- `gamma = 0.99`
- `gae_lambda = 0.95`
- Seed: `0`

Models are saved automatically under `models/`.

For a quick smoke test:

```bash
python run_training.py --sigmas 0.0 0.1 0.2 --total-timesteps 5000 --n-steps 512 --n-epochs 2
```

The smaller rollout length and epoch count make the smoke test much faster than
the full research configuration. The training runner also stops at the requested
timestep budget instead of letting Stable-Baselines3 round up to the next full
rollout.

## Run Analysis

After training, run:

```bash
python run_analysis.py
```

This writes:

- `results/ppo_evaluation.csv`
- `results/gae_bias.csv`
- `results/fit_statistics.csv`
- `results/model_residuals.csv`

For faster analysis while debugging:

```bash
python run_analysis.py --sigmas 0.0 0.1 0.2 --eval-episodes 3 --gae-episodes 5
```

Use at least two nonzero sigma values if you want the log-log bias plot to be
generated during a smoke test.

## Generate Figures

`run_analysis.py` generates all plots automatically. You can regenerate figures
from existing result CSVs with:

```bash
python make_plots.py
```

Figures are saved in `figures/` as both PNG and PDF:

- `fig_advantage_bias_vs_sigma`
- `fig_ppo_return_vs_sigma`
- `fig_advantage_bias_vs_return`
- `fig_loglog_bias`
- `fig_residual_analysis`
- `fig_training_curves`

## Original Analytical Reproduction

The original paper-style numerical experiment is preserved:

```bash
python run_experiment.py
```

The analytical script still writes its original text and CSV outputs:

- `results_table.csv`
- `regression_stats.txt`
- `critical_threshold.txt`

The pre-rendered analytical figures from the earlier repository remain in
`figures/`. The updated `make_plots.py` is now dedicated to the PPO analysis
CSVs produced by `run_analysis.py`.

## Citation

If you use this code, please cite:

```bibtex
@article{patil2026quadratic,
  title   = {A Quadratic Bound on PPO Advantage Bias Under UAV Turbulence},
  author  = {Sahil, Nasiruddin},
  journal = {International Journal of Engineering Research \& Technology (IJERT)},
  volume  = {15},
  number  = {05},
  year    = {2026}
}
```
