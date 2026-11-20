"""Synthetic covariate shift with known density ratio, to verify the guarantees.

Patient covariate x ~ N(mu, I_d)  (source mu = 0, target mu = shift * e_1).
Patient "hardness"  h = gamma * x_1 + noise,  loss curve  L(lambda) = (1 - lambda)^{exp(-h)}.
The conditional law of the loss given x is the same in both domains (covariate shift),
target patients are harder, so standard CRC under-controls the risk on the target.
"""
from __future__ import annotations

import numpy as np

from .weights import gaussian_true_weights


def make_domain(n, d, mu, lambdas, gamma=0.5, noise=0.3, rng=None):
    rng = np.random.default_rng(rng)
    x = rng.normal(size=(n, d)) + mu
    h = gamma * x[:, 0] + noise * rng.normal(size=n)
    k = np.exp(-h)
    L = (1.0 - lambdas[None, :]) ** k[:, None]
    u = h + 0.5 * rng.normal(size=n)          # noisy uncertainty proxy
    return x, L, u


def make_problem(n_cal=100, n_test=2000, d=8, shift=1.0, n_lambda=501, seed=0, **kw):
    lambdas = np.linspace(0, 1, n_lambda)
    mu_s = np.zeros(d)
    mu_t = np.zeros(d)
    mu_t[0] = shift
    rng = np.random.default_rng(seed)
    xs, Ls, us = make_domain(n_cal, d, mu_s, lambdas, rng=rng, **kw)
    xt, Lt, ut = make_domain(n_test, d, mu_t, lambdas, rng=rng, **kw)
    return {
        "lambdas": lambdas,
        "x_cal": xs, "L_cal": Ls, "u_cal": us,
        "x_test": xt, "L_test": Lt, "u_test": ut,
        "w_cal_true": gaussian_true_weights(xs, mu_s, mu_t),
        "w_test_true": gaussian_true_weights(xt, mu_s, mu_t),
    }
