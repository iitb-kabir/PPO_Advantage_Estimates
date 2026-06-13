# PPO Advantage Bias Under UAV Turbulence

This repository studies how PPO advantage estimates and control performance
degrade as aerodynamic turbulence intensity `sigma` increases, for a linear 2D
UAV control task.

- A closed-form **LQR / Riccati** baseline confirms the theoretical law
  `bias = C sigma^2` almost exactly.
- A learned **Stable-Baselines3 PPO** pipeline tests the same law with neural
  policies, across multiple random seeds, with mean +/- 95% CI reporting.
- A **turbulence curriculum** experiment tests the square-root training schedule
  implied by the theory.
- An optional **colored (Dryden) turbulence** model checks that the law survives
  temporally-correlated gusts.

---

## 1. Quickstart (TL;DR)

```bash
# 1. get the code
git clone https://github.com/iitb-kabir/PPO_Advantage_Estimates.git
cd PPO_Advantage_Estimates

# 2. create an environment and install dependencies (see Section 2 for details)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. verify the setup with the fast seed test (a few minutes, see Section 4)
python run_training.py --sigmas 0.0 0.1 0.2 --seeds 0 1 \
    --total-timesteps 5000 --n-steps 512 --n-epochs 2
python run_analysis.py --sigmas 0.0 0.1 0.2 --seeds 0 1 \
    --eval-episodes 3 --gae-episodes 5

# 4. run the full experiments (hours; see Section 5)
python run_training.py
python run_analysis.py
python run_curriculum.py --sigma-max 0.5
```

---

## 2. Setup from scratch

### Requirements

- **Python 3.10 or newer**
- The packages in `requirements.txt`:
  `numpy`, `scipy`, `matplotlib`, `pandas`, `torch`, `gymnasium`,
  `stable-baselines3[extra]`, `tqdm`
- A **GPU is optional**. The code uses `device="auto"` (see Section 3), so it
  runs on CPU-only machines without any changes.

### Option A — `venv` + pip (simplest)

```bash
cd PPO_Advantage_Estimates
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — conda

```bash
cd PPO_Advantage_Estimates
conda create -n ppo-uav python=3.10 -y
conda activate ppo-uav
pip install -r requirements.txt
```

### Check the install

```bash
python -c "import torch, gymnasium, stable_baselines3 as sb3; \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| sb3', sb3.__version__)"
```

This should print version numbers without errors. `cuda True` means a GPU will
be used; `cuda False` is fine and the code will use the CPU automatically.

> **Note on PyTorch / GPU.** `pip install -r requirements.txt` installs the
> default PyTorch build. If you have an NVIDIA GPU and want CUDA acceleration but
> `torch.cuda.is_available()` prints `False`, install the matching CUDA build of
> PyTorch from <https://pytorch.org/get-started/locally/> and re-run the check.

---

## 3. Hardware / device

`config.py` sets `PPOConfig.device = "auto"`:

- with a GPU, training and analysis run on CUDA;
- without a GPU, everything falls back to CPU automatically.

You do not need to edit anything. Full training is heavier on CPU but still
completes (the task is small: two 64x64 MLPs). To force a specific device, set
`device="cpu"` or `device="cuda"` in `config.py`.

---

## 4. Verify your setup: the seed test (do this first)

This is a tiny end-to-end run that exercises **the entire pipeline** — multi-seed
training, evaluation, GAE-bias measurement, scaling fits, the neural-critic
check, and figure generation — in a few minutes. Run it before the full
experiments to confirm the environment is correct.

```bash
# trains 3 sigma x 2 seeds = 6 tiny models
python run_training.py --sigmas 0.0 0.1 0.2 --seeds 0 1 \
    --total-timesteps 5000 --n-steps 512 --n-epochs 2

# runs the full analysis on those models
python run_analysis.py --sigmas 0.0 0.1 0.2 --seeds 0 1 \
    --eval-episodes 3 --gae-episodes 5
```

It is expected to finish without errors and to create:

- models: `models/ppo_uav_sigma_0_00_seed0.zip`, `..._seed1.zip`, etc.
- result tables in `results/` (see Section 6)
- figures in `figures/`

The numbers from this tiny run are **not** meaningful (5,000 steps is far too
short to learn) — the point is only to confirm the pipeline runs end to end.

> The seed test uses two nonzero sigma values (`0.1` and `0.2`) so the log-log
> bias plot can be generated. With a single nonzero sigma that plot is skipped.

You can run an even faster curriculum check the same way:

```bash
python run_curriculum.py --sigma-max 0.2 --seeds 0 1 \
    --total-timesteps 5000 --n-steps 512 --n-epochs 2 --episodes 10
```

---

## 5. Full experiments

> **Important:** use the *same* `--seeds` for training and analysis. If you omit
> `--seeds`, both default to `0 1 2 3 4`.

### 5.1 Train the PPO sweep

```bash
python run_training.py
```

This trains one model per (sigma, seed): `len(sigmas) * len(seeds)` models.
Defaults (in `config.py`):

| Setting | Value |
| --- | --- |
| sigma ladder | `0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00` |
| seeds (shared across all sigma) | `0, 1, 2, 3, 4` |
| timesteps per (sigma, seed) | `1,000,000` |
| policy / value network | `64 -> 64`, Tanh |
| `gamma`, `gae_lambda` | `0.99`, `0.95` |

The same seed set is shared across every sigma, so the seed is **decoupled from
sigma** and results can be reported as mean +/- 95% CI. Models are saved as
`models/ppo_uav_sigma_<sigma>_seed<seed>.zip`.

Use fewer seeds if compute is limited:

```bash
python run_training.py --seeds 0 1 2
```

### 5.2 Run the analysis

```bash
python run_analysis.py
```

This evaluates every model, measures the GAE bias (nominal and matched policy —
see Section 6), fits the scaling laws, validates the bound against the learned
critic, and regenerates all figures.

### 5.3 Turbulence curriculum experiment

Theorem 1 implies a square-root turbulence schedule: because bias scales as
`C sigma^2`, ramping `sigma(p) = sigma_max * sqrt(p)` makes the bias grow
linearly in training progress `p`. This experiment trains PPO with the
`constant` (fixed-sigma baseline), `linear`, and `sqrt` schedules and evaluates
them at the deployment level `sigma_max`:

```bash
python run_curriculum.py --sigma-max 0.5
```

Re-run only the evaluation/plot on already-trained curriculum models with
`--eval-only`. Outputs `results/curriculum_eval.csv` and
`figures/fig_curriculum_comparison`.

### 5.4 Colored (Dryden) turbulence variant

By default gusts are i.i.d. Gaussian (`turbulence_type = "white"`). To use a
first-order colored gust that approximates the Dryden lateral spectrum, edit
`UAVDynamicsConfig` in `config.py`:

```python
turbulence_type: str = "dryden"   # was "white"
dryden_tau: float = 1.0           # gust correlation time in seconds
```

Then re-run training and analysis. The quadratic bias law is predicted to
survive (only the propagation gain `kappa` changes). Tip: keep your white-noise
results by copying `results/` and `models/` aside before switching, so you can
compare the fitted exponents.

---

## 6. Outputs

### Result tables (`results/`)

Each analysis writes a **seed-aggregated** table plus a per-seed `*_raw.csv`
companion (one row per seed, used for the confidence intervals).

| File | Contents |
| --- | --- |
| `ppo_evaluation.csv` | return / success per sigma, mean +/- 95% CI across seeds |
| `gae_bias.csv` | **nominal-policy** bias (theory test): the sigma=0 policy run under increasing turbulence. Includes CI, median, IQR, and the fraction of diverged episodes |
| `gae_bias_matched.csv` | **matched-policy** bias (deployment test): the policy trained at each sigma, evaluated at that sigma, so bias and return/success describe the *same* controller |
| `fit_statistics.csv` | quadratic and free power-law fits with consistent log-space R^2 and AIC, plus a per-seed exponent mean +/- 95% CI |
| `theory_comparison.csv` | empirical vs Theorem-1 coefficient `C` (and the ratio), `||P||`, closed-loop spectral radius, `kappa` |
| `critical_threshold.csv` | `sigma*(f)` for `f` in `{0.05, 0.10, 0.25, 0.50}` |
| `critic_hessian.csv` | learned-critic Hessian spectral norm vs LQR `2||P||`, and the stable-regime (small-sigma) bias exponent |
| `curriculum_eval.csv` | per-schedule return / success at `sigma_max`, mean +/- 95% CI |
| `model_residuals.csv` | per-sigma residuals for the fitted models |

### Figures (`figures/`, saved as both PNG and PDF)

`run_analysis.py` generates these automatically; regenerate them from existing
CSVs with `python make_plots.py`.

- `fig_advantage_bias_vs_sigma` — bias vs sigma with 95% CI error bars
- `fig_ppo_return_vs_sigma` — return vs sigma with 95% CI
- `fig_advantage_bias_vs_return`
- `fig_loglog_bias`
- `fig_residual_analysis`
- `fig_robust_bias` — mean vs median+IQR, plus the diverged-episode fraction
- `fig_curriculum_comparison` — only if curriculum results exist
- `fig_training_curves`

---

## 7. Useful command-line flags

`run_training.py`:
`--sigmas`, `--seeds`, `--total-timesteps`, `--n-steps`, `--batch-size`,
`--n-epochs`

`run_analysis.py`:
`--sigmas`, `--seeds`, `--eval-episodes`, `--gae-episodes`, `--policy-sigma`,
`--skip-evaluation`, `--skip-gae`, `--skip-matched-gae`, `--skip-critic-hessian`,
`--skip-plots`

`run_curriculum.py`:
`--schedules`, `--seeds`, `--sigma-max`, `--total-timesteps`, `--n-steps`,
`--batch-size`, `--n-epochs`, `--episodes`, `--eval-only`

---

## 8. Repository structure

```text
.
|-- envs/
|   |-- __init__.py
|   `-- uav_env.py          # Gymnasium env (white + Dryden turbulence)
|-- training/
|   |-- __init__.py
|   |-- train_ppo.py        # multi-seed PPO training
|   `-- curriculum.py       # constant/linear/sqrt curriculum training + eval
|-- analysis/
|   |-- __init__.py
|   |-- evaluate_models.py  # multi-seed evaluation + CI aggregation
|   |-- compute_gae_bias.py # robust GAE-bias (nominal + matched policy)
|   |-- fit_models.py       # log-space fits, AIC, per-seed exponent CI
|   |-- lqr_reference.py    # closed-form Riccati/LQR bound constants
|   `-- critic_hessian.py   # validate the bound against the learned critic
|-- figures/
|-- models/
|-- results/
|-- config.py               # all settings (sigmas, seeds, PPO, turbulence, curriculum)
|-- run_training.py
|-- run_analysis.py
|-- run_curriculum.py
|-- run_experiment.py       # original analytical LQR reproduction
|-- make_plots.py
|-- README.md
`-- requirements.txt
```

---

## 9. Environment definition

The Gymnasium environment is `envs/uav_env.py::UAVTurbulenceEnv`.

```text
state   s = [x, vx, y, vy]
action  a = [ax, ay]
dynamics  s_{t+1} = A s_t + B a_t + E w_t,   w_t ~ N(0, sigma^2)
reward    r = -(s^T Q s + a^T R a),  Q = I,  R = 0.1 I

A = [[1, dt, 0,  0 ],     B = [[0,  0 ],     E = [0, 0, 0, 1]
     [0, 1,  0,  0 ],          [dt, 0 ],
     [0, 0,  1,  dt],          [0,  0 ],
     [0, 0,  0,  1 ]]          [0,  dt]]
```

`dt = 0.1 s`, episode length `200` steps. With `turbulence_type="dryden"` the
gust `w_t` becomes a first-order colored process with correlation time
`dryden_tau` and the same stationary standard deviation `sigma`.

---

## 10. Original analytical reproduction

The original NumPy/SciPy LQR Monte Carlo experiment is preserved:

```bash
python run_experiment.py
```

It writes `results_table.csv`, `regression_stats.txt`, and
`critical_threshold.txt`. (Note: this legacy script uses rollout length `T=20`;
the PPO pipeline and `analysis/lqr_reference.py` use the full episode length
`T=200`.)

---

## 11. Citation

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
