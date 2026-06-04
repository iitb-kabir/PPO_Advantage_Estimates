"""
run_experiment.py
=================
Reproduces all numerical results in:

  "On the Quadratic Degradation of PPO Advantage Estimates Under
   Non-Stationary Aerodynamic Disturbances: A Theoretical Bound and
   Empirical Validation"

Requirements: numpy, scipy  (pip install numpy scipy)

Output files:
  results_table.csv      -- Table I in the paper
  regression_stats.txt   -- Quadratic regression statistics
  critical_threshold.txt -- sigma* values

Usage:
  python run_experiment.py
  python run_experiment.py --seed 0 --n_trials 3000
"""

import numpy as np
import csv
import argparse
from scipy import stats

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed',     type=int,   default=0,    help='Random seed (paper uses 0)')
parser.add_argument('--n_trials', type=int,   default=3000, help='Monte Carlo trials per sigma')
parser.add_argument('--T',        type=int,   default=20,   help='Rollout length')
parser.add_argument('--dt',       type=float, default=0.1,  help='Timestep (seconds)')
parser.add_argument('--gamma',    type=float, default=0.99, help='Discount factor')
parser.add_argument('--lam',      type=float, default=0.95, help='GAE lambda')
args = parser.parse_args()

np.random.seed(args.seed)
T   = args.T
dt  = args.dt
g   = args.gamma
lam = args.lam
n   = args.n_trials

SIGMAS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]

print(f"Configuration: seed={args.seed}, n_trials={n}, T={T}, dt={dt}, gamma={g}, lambda={lam}")
print("="*70)

# -----------------------------------------------------------------------------
# SYSTEM MATRICES  (2D UAV: state = [x, vx, y, vy], action = [ax, ay])
# -----------------------------------------------------------------------------
A_sys = np.array([
    [1, dt, 0,  0 ],
    [0, 1,  0,  0 ],
    [0, 0,  1,  dt],
    [0, 0,  0,  1 ]
], dtype=float)

B_sys = np.array([
    [0,  0 ],
    [dt, 0 ],
    [0,  0 ],
    [0,  dt]
], dtype=float)

E_dist = np.array([0, 0, 0, 1], dtype=float)   # disturbance enters vy

Q = np.eye(4)
R = 0.1 * np.eye(2)

# -----------------------------------------------------------------------------
# DISCRETE-TIME ALGEBRAIC RICCATI EQUATION (DARE) solver
# -----------------------------------------------------------------------------
def solve_dare(A, B, Q, R, gamma, max_iter=2000, tol=1e-12):
    P = np.eye(A.shape[0])
    n_iter = max_iter
    for i in range(max_iter):
        BtPB = B.T @ P @ B
        P_new = Q + gamma * A.T @ P @ A \
              - gamma**2 * A.T @ P @ B @ np.linalg.inv(R + gamma * BtPB) @ B.T @ P @ A
        if np.max(np.abs(P_new - P)) < tol:
            n_iter = i + 1
            print(f"  DARE converged in {n_iter} iterations")
            P = P_new
            break
        P = P_new
    return P, n_iter

print("Solving DARE for LQR value function...")
P, dare_iters = solve_dare(A_sys, B_sys, Q, R, g)
P_spec = np.linalg.norm(P, ord=2)
print(f"  P diagonal: {np.diag(P).round(4)}")
print(f"  P spectral norm: {P_spec:.4f}")

# LQR gain
K = np.linalg.inv(R + g * B_sys.T @ P @ B_sys) @ (g * B_sys.T @ P @ A_sys)
A_cl = A_sys - B_sys @ K   # closed-loop matrix
rho = np.max(np.abs(np.linalg.eigvals(A_cl)))
print(f"  Closed-loop spectral radius: {rho:.4f}  ({'stable' if rho < 1 else 'UNSTABLE'})")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def value_fn(s):
    """True LQR value function V*(s) = -s^T P s"""
    return -s @ P @ s

def lqr_action(s):
    """LQR policy: a = -K s"""
    return -K @ s

def step_nominal(s, a):
    return A_sys @ s + B_sys @ a

def step_disturbed(s, a, sigma):
    w = np.random.normal(0, sigma)
    return A_sys @ s + B_sys @ a + E_dist * w

def reward(s, a):
    return -(s @ Q @ s + a @ R @ a)

def compute_gae(rewards, values):
    T_ = len(rewards)
    advantages = np.zeros(T_)
    gae = 0.0
    for t in reversed(range(T_)):
        next_v = values[t + 1] if t + 1 < T_ else 0.0
        delta = rewards[t] + g * next_v - values[t]
        gae = delta + g * lam * gae
        advantages[t] = gae
    return advantages

def run_trial(sigma):
    s0 = np.random.randn(4) * 0.5

    rn, vn = [], []
    s = s0.copy()
    for _ in range(T):
        a = lqr_action(s)
        rn.append(reward(s, a))
        vn.append(value_fn(s))
        s = step_nominal(s, a)

    rd, vd = [], []
    s = s0.copy()
    for _ in range(T):
        a = lqr_action(s)
        rd.append(reward(s, a))
        vd.append(value_fn(s))
        s = step_disturbed(s, a, sigma)

    A_nom = compute_gae(rn, vn)
    A_dis = compute_gae(rd, vd)
    return np.mean(np.abs(A_nom - A_dis))

# -----------------------------------------------------------------------------
# MAIN EXPERIMENT
# -----------------------------------------------------------------------------
print(f"\nRunning {n} trials x {len(SIGMAS)} sigma levels = {n*len(SIGMAS):,} total trial pairs")
print("-"*70)

results = {}
for sigma in SIGMAS:
    errors = np.array([run_trial(sigma) for _ in range(n)])
    mean_e = float(np.mean(errors))
    sem_e  = float(np.std(errors) / np.sqrt(n))
    results[sigma] = (mean_e, sem_e)
    print(f"  sigma={sigma:.2f}  E[DeltaA]={mean_e:.5f}  +/-{1.96*sem_e:.5f} (95%CI)")

# -----------------------------------------------------------------------------
# COMPUTE NOMINAL STATE NORM  r_bar = E[||s_t||]  (from data, not hardcoded)
# -----------------------------------------------------------------------------
state_norms = []
for _ in range(2000):
    s = np.random.randn(4) * 0.5
    for _ in range(T):
        state_norms.append(np.linalg.norm(s))
        s = step_nominal(s, lqr_action(s))
r_bar = float(np.mean(state_norms))
print(f"\nNominal state norm r_bar = E[||s||] = {r_bar:.4f} (computed from rollouts)")

# -----------------------------------------------------------------------------
# THEORETICAL BOUND  (QUADRATIC in sigma)
# C = ||P||_2 * kappa / (1 - (gamma*lambda)^2),  kappa = sum_k ||A_cl^k E||^2 contribution
# Leading-order second-moment constant.
# -----------------------------------------------------------------------------
# E[||delta_{t+k}||^2] = ||A_cl^k E||^2 * sigma^2
kappa = sum(np.linalg.norm(np.linalg.matrix_power(A_cl, k) @ E_dist)**2 for k in range(T))
C_theoretical = 2 * P_spec * kappa * T / (1 - (g*lam)**2)
print(f"\nTheoretical bound constant (quadratic) C:")
print(f"  C = 2*||P||_2 * kappa * T / (1-(gamma*lambda)^2)")
print(f"    = 2 * {P_spec:.4f} * {kappa:.4f} * {T} / (1 - {(g*lam)**2:.4f})")
print(f"    = {C_theoretical:.4f}")

# -----------------------------------------------------------------------------
# QUADRATIC REGRESSION:  E[DeltaA] = C_emp * sigma^2
# -----------------------------------------------------------------------------
sigmas_arr = np.array(SIGMAS)
errors_arr = np.array([results[s][0] for s in SIGMAS])

# Fit through origin on sigma^2
C_emp = float(np.sum(sigmas_arr**2 * errors_arr) / np.sum(sigmas_arr**4))
pred = C_emp * sigmas_arr**2
ss_res = np.sum((errors_arr - pred)**2)
ss_tot = np.sum((errors_arr - errors_arr.mean())**2)
r2 = 1 - ss_res/ss_tot

# Regression on sigma^2 for p-value
slope, intercept, r_val, p_val, se = stats.linregress(sigmas_arr**2, errors_arr)

# Log-log exponent check
mask = sigmas_arr > 0
ls, li, lr, lp, _ = stats.linregress(np.log(sigmas_arr[mask]), np.log(errors_arr[mask]))

print(f"\nQuadratic Regression: E[DeltaA] = {C_emp:.4f} * sigma^2")
print(f"  R^2 (through origin)   = {r2:.6f}")
print(f"  slope on sigma^2       = {slope:.4f}")
print(f"  p-value                = {p_val:.3e}")
print(f"  log-log exponent       = {ls:.4f}  (expect ~2.0)")
print(f"  C_emp  = {C_emp:.4f}")
print(f"  C_theo = {C_theoretical:.4f}")
print(f"  Ratio  = {C_emp/C_theoretical:.5f}")

# -----------------------------------------------------------------------------
# CRITICAL THRESHOLD:  f*A_scale = C_emp * sigma^2  =>  sigma* = sqrt(f*A/C_emp)
# -----------------------------------------------------------------------------
nom_advs = []
for _ in range(1000):
    s0 = np.random.randn(4) * 0.5
    rn, vn = [], []
    s = s0.copy()
    for _ in range(T):
        a = lqr_action(s)
        rn.append(reward(s, a))
        vn.append(value_fn(s))
        s = step_nominal(s, a)
    nom_advs.extend(np.abs(compute_gae(rn, vn)).tolist())

A_scale = float(np.mean(nom_advs))
print(f"\nTypical nominal advantage scale E[|A^GAE|] = {A_scale:.5f}")
for f in [0.10, 0.25, 0.50]:
    sigma_star = np.sqrt(f * A_scale / C_emp)
    print(f"  sigma*(f={f:.0%}) = {sigma_star:.4f} m/s")

# -----------------------------------------------------------------------------
# SAVE OUTPUTS
# -----------------------------------------------------------------------------
with open('results_table.csv', 'w', newline='') as fcsv:
    w = csv.writer(fcsv)
    w.writerow(['sigma', 'mean_error', 'sem', 'ci_95', 'quad_fit', 'norm'])
    max_e = errors_arr[-1]
    for sigma in SIGMAS:
        m, se_v = results[sigma]
        w.writerow([sigma, round(m,5), round(se_v,5), round(1.96*se_v,5),
                    round(C_emp*sigma**2,5), round(m/max_e,3)])
print("\nSaved: results_table.csv")

with open('regression_stats.txt', 'w') as f:
    f.write(f"Quadratic model: E[DeltaA] = {C_emp:.6f} * sigma^2\n")
    f.write(f"R^2 (origin)     = {r2:.6f}\n")
    f.write(f"slope on sigma^2 = {slope:.6f}\n")
    f.write(f"p-value          = {p_val:.6e}\n")
    f.write(f"log-log exponent = {ls:.6f}\n")
    f.write(f"C_emp            = {C_emp:.6f}\n")
    f.write(f"C_theo           = {C_theoretical:.6f}\n")
    f.write(f"Ratio            = {C_emp/C_theoretical:.8f}\n")
    f.write(f"P spectral norm  = {P_spec:.6f}\n")
    f.write(f"closed-loop rho  = {rho:.6f}\n")
    f.write(f"DARE iterations  = {dare_iters}\n")
    f.write(f"r_bar            = {r_bar:.6f}\n")
print("Saved: regression_stats.txt")

with open('critical_threshold.txt', 'w') as f:
    f.write(f"Nominal advantage scale: {A_scale:.6f}\n")
    for frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.50]:
        sigma_star = np.sqrt(frac * A_scale / C_emp)
        f.write(f"sigma*(f={frac:.2f}) = {sigma_star:.6f} m/s\n")
print("Saved: critical_threshold.txt")

print("\n" + "="*70)
print("EXPERIMENT COMPLETE.")
print("="*70)
