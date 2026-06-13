"""Closed-form LQR reference quantities for the quadratic-bias bound.

Solves the discrete-time algebraic Riccati equation for the UAV double-integrator
and returns the Riccati matrix ``P``, the LQR gain ``K``, the closed-loop matrix
``A_cl = A - B K``, and the constants appearing in Theorem 1
(``C = 2||P||_2 * kappa * T / (1 - (gamma*lambda)^2)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CONFIG, ExperimentConfig


@dataclass(frozen=True)
class LQRReference:
    """Closed-form LQR/Riccati quantities used by the bound."""

    P: np.ndarray
    K: np.ndarray
    A_cl: np.ndarray
    p_spectral_norm: float
    closed_loop_rho: float
    kappa: float
    c_theoretical: float
    dare_iterations: int


def solve_dare(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    gamma: float,
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, int]:
    """Iterate the discounted DARE recursion to a fixed point."""

    P = np.eye(A.shape[0])
    n_iter = max_iter
    for i in range(max_iter):
        BtPB = B.T @ P @ B
        P_new = (
            Q
            + gamma * A.T @ P @ A
            - gamma**2 * A.T @ P @ B @ np.linalg.inv(R + gamma * BtPB) @ B.T @ P @ A
        )
        if np.max(np.abs(P_new - P)) < tol:
            n_iter = i + 1
            P = P_new
            break
        P = P_new
    return P, n_iter


def lqr_reference(config: ExperimentConfig = CONFIG, T: int | None = None) -> LQRReference:
    """Compute LQR/Riccati reference quantities for the configured dynamics."""

    dyn = config.dynamics
    A = dyn.a_matrix.astype(float)
    B = dyn.b_matrix.astype(float)
    Q = dyn.q_matrix.astype(float)
    R = dyn.r_matrix.astype(float)
    E = dyn.e_matrix.astype(float)
    gamma = config.ppo.gamma
    lam = config.ppo.gae_lambda
    horizon = T or dyn.episode_length

    P, n_iter = solve_dare(A, B, Q, R, gamma)
    p_spec = float(np.linalg.norm(P, ord=2))
    K = np.linalg.inv(R + gamma * B.T @ P @ B) @ (gamma * B.T @ P @ A)
    A_cl = A - B @ K
    rho = float(np.max(np.abs(np.linalg.eigvals(A_cl))))

    if E.ndim == 1:
        kappa = float(
            sum(np.linalg.norm(np.linalg.matrix_power(A_cl, k) @ E) ** 2 for k in range(horizon))
        )
    else:
        kappa = float(
            sum(
                np.linalg.norm(np.linalg.matrix_power(A_cl, k) @ E, ord=2) ** 2
                for k in range(horizon)
            )
        )
    c_theoretical = 2.0 * p_spec * kappa * horizon / (1.0 - (gamma * lam) ** 2)

    return LQRReference(
        P=P,
        K=K,
        A_cl=A_cl,
        p_spectral_norm=p_spec,
        closed_loop_rho=rho,
        kappa=kappa,
        c_theoretical=c_theoretical,
        dare_iterations=n_iter,
    )
