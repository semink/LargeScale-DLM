from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy import sparse as sp
import numpy as np


def log_evidence(alpha, beta, T, s, V, X, M):
    N, D = T.shape
    gamma = np.sum(beta)
    pi = beta / gamma
    sigma = 1 / alpha + 1 / gamma * (s ** 2)
    Hp = M @ pi
    logp = -N * D / 2 * np.log(2 * np.pi) \
           - N / 2 * np.sum(np.log(sigma)) \
           - 1 / 2 * ((T - Hp @ X) @ V @ sp.diags(1 / sigma) @ V.T @ (T - Hp @ X).T).diagonal().sum()
    if np.isnan(logp): breakpoint()
    return logp


def negative_log_evidence(x, T, s, V, X, M):
    alpha, beta = x[0], x[1:]
    return -log_evidence(alpha, beta, T, s, V, X, M)


def exclude_nan(X, T):
    x_valid_idx = np.all(~np.isnan(X), axis=0)
    t_valid_idx = np.all(~np.isnan(T), axis=0)
    valid_idx = x_valid_idx & t_valid_idx
    return X[:, valid_idx], T[:, valid_idx]


def get_optimal_x(x0, *args, bounds, method='L-BFGS-B', options={'maxiter': 500, 'tol': 1e-8, 'disp': True}):
    res = minimize(negative_log_evidence, x0=x0, args=args,
                   method=method,
                   bounds=bounds, options={'maxiter': options['maxiter'], 'disp': options['disp']},
                   tol=options['tol'],
                   )
    return res


def get_optimal_H(X, T, M):
    params = dict()
    if M is not None:
        bounds = Bounds((M.shape[2] + 1) * [1e-7], (M.shape[2] + 1) * [np.inf])
        x0 = np.ones((M.shape[2] + 1))
        N = X.shape[0]
        _, s, Vh = np.linalg.svd(X, full_matrices=False)
        V = Vh.T

        res = get_optimal_x(x0, T, s, V, X, M, bounds=bounds)  # first try

        if not res.success: res = get_optimal_x(np.random.rand(M.shape[2] + 1), T, s, V, X, M,
                                                bounds=bounds)  # second try
        if not res.success: res = get_optimal_x(x0, T, s, V, X, M, bounds=bounds, method='TNC')  # second try

        alpha, beta = res.x[0], res.x[1:]
        gamma = np.sum(beta)
        pi = beta / gamma
        Hp = M @ pi
        success = res.success
        H = (alpha * T @ X.T + gamma * Hp) @ np.linalg.pinv(alpha * X @ X.T + gamma * np.eye(N))
        params['alpha'], params['beta'], params['gamma'], params['pi'] = alpha, beta, gamma, pi

    else:
        H = (T @ X.T) @ np.linalg.pinv(X @ X.T)
        success = True
    return H, success, params