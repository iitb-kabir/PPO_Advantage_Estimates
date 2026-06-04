"""
make_plots.py
=============
Generates publication figures for the quadratic-degradation paper.
Outputs (300 dpi, IEEE-column width):
  fig_loglog.pdf      -- log-log error vs sigma, slope annotated
  fig_mainfit.pdf     -- error + 95% CI, quadratic fit vs linear fit
  fig_residuals.pdf   -- linear vs quadratic residuals
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# --- Data (means over n=3000 trial pairs; from run_experiment.py) -------------
sigma = np.array([0.00,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.80,1.00])
err   = np.array([0.00000,0.09668,0.31322,0.67442,1.18369,2.66683,
                  4.68683,7.41101,10.33991,18.81844,29.02556])
ci95  = np.array([0.00000,0.00235,0.00704,0.01508,0.02635,0.05801,
                  0.10047,0.16060,0.21708,0.39484,0.64810])

# --- Fits ---------------------------------------------------------------------
C_emp = np.sum(sigma**2 * err) / np.sum(sigma**4)
quad  = lambda x: C_emp * x**2
sl, ic, r, p, se = stats.linregress(sigma, err)
lin   = lambda x: sl*x + ic
m = sigma > 0
ll_slope, ll_int, ll_r, _, _ = stats.linregress(np.log(sigma[m]), np.log(err[m]))

def r2(y, yhat):
    ss_res = np.sum((y-yhat)**2); ss_tot = np.sum((y-y.mean())**2)
    return 1 - ss_res/ss_tot
r2_quad = r2(err, quad(sigma))
r2_lin  = r2(err, lin(sigma))

# --- Style --------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": True, "legend.framealpha": 0.92,
    "legend.edgecolor": "0.8", "figure.dpi": 300,
})
COL = 3.5

# =============================================================================
# FIG 1: LOG-LOG  (legend in lower-right empty region, framed)
# =============================================================================
fig, ax = plt.subplots(figsize=(COL, 2.7))
ax.loglog(sigma[m], err[m], 'o', ms=5, color='#1f3b73', label='Measured', zorder=3)
xs = np.linspace(sigma[m].min(), sigma.max(), 200)
ax.loglog(xs, np.exp(ll_int)*xs**ll_slope, '-', lw=1.5, color='#c0392b',
          label=f'Power-law fit (slope = {ll_slope:.2f})')
ref = err[-1]*(xs/sigma[-1])**2
ax.loglog(xs, ref, '--', lw=1.0, color='gray', label='Slope = 2 reference')
ax.set_xlabel(r'Turbulence intensity $\sigma$ (m/s)')
ax.set_ylabel(r'GAE bias $\mathbb{E}[\bar\Delta(\sigma)]$')
ax.legend(loc='lower right', fontsize=7.5)   # empty corner here
ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.6)
fig.tight_layout(pad=0.4)
fig.savefig('fig_loglog.pdf', bbox_inches='tight')
plt.close(fig)

# =============================================================================
# FIG 2: MAIN FIT  -- annotation moved into open upper-left area, away from lines
# =============================================================================
fig, ax = plt.subplots(figsize=(COL, 2.9))
xs = np.linspace(0, 1.0, 300)
ax.plot(xs, quad(xs), '-', lw=1.6, color='#1f3b73',
        label=fr'Quadratic fit $C\sigma^2$, $R^2={r2_quad:.4f}$')
ax.plot(xs, lin(xs), '--', lw=1.3, color='#c0392b',
        label=fr'Linear fit, $R^2={r2_lin:.3f}$')
ax.axhline(0, color='k', lw=0.5)
ax.errorbar(sigma, err, yerr=ci95, fmt='o', ms=4, color='black',
            ecolor='gray', elinewidth=0.9, capsize=2,
            label='Measured (95% CI)', zorder=4)
# Headroom so legend (upper-left) and annotation don't touch curves
ax.set_ylim(-6.5, 33)
ax.set_xlim(-0.02, 1.02)
# Annotation box placed in the empty lower-MIDDLE region, arrow to the intercept.
ax.annotate('Linear fit reaches\nunphysical bias < 0\nat ' r'$\sigma=0$ (int. $=-3.49$)',
            xy=(0.0, ic), xytext=(0.40, -6.0),
            fontsize=6.8, color='#c0392b', ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#c0392b', lw=0.6),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8,
                            connectionstyle='arc3,rad=0.2'))
ax.set_xlabel(r'Turbulence intensity $\sigma$ (m/s)')
ax.set_ylabel(r'GAE bias $\mathbb{E}[\bar\Delta(\sigma)]$')
ax.legend(loc='upper left', fontsize=7)
ax.grid(True, ls=':', lw=0.4, alpha=0.6)
fig.tight_layout(pad=0.4)
fig.savefig('fig_mainfit.pdf', bbox_inches='tight')
plt.close(fig)

# =============================================================================
# FIG 3: RESIDUALS -- legend moved OUTSIDE plot (below), extra y-headroom
# =============================================================================
fig, ax = plt.subplots(figsize=(COL, 2.7))
res_lin  = err - lin(sigma)
res_quad = err - quad(sigma)
ax.plot(sigma, res_lin, 's--', ms=4, lw=1.0, color='#c0392b',
        label='Linear model residuals')
ax.plot(sigma, res_quad, 'o-', ms=4, lw=1.0, color='#1f3b73',
        label='Quadratic model residuals')
ax.axhline(0, color='k', lw=0.6)
pad = 0.15*(res_lin.max()-res_lin.min())
ax.set_ylim(res_lin.min()-pad, res_lin.max()+pad)
ax.set_xlabel(r'Turbulence intensity $\sigma$ (m/s)')
ax.set_ylabel('Residual')
# Legend placed below the axes so it never sits on the data
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28),
          ncol=2, fontsize=7.5, frameon=False)
ax.grid(True, ls=':', lw=0.4, alpha=0.6)
fig.tight_layout(pad=0.4)
fig.savefig('fig_residuals.pdf', bbox_inches='tight')
plt.close(fig)

print("Saved: fig_loglog.pdf, fig_mainfit.pdf, fig_residuals.pdf")
