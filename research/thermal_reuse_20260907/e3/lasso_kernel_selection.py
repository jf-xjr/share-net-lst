"""Input-only adaptation of the LASSO selection step in Yoo et al. (2022).

This is kernel selection for the subsequent RF/LLF, not the final reconstruction.
The sole response is the currently observed coarse thermal field. The caller
must establish finite, matched historical kernels and a fixed spatial fold map.
"""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ALPHAS = np.arange(1, 101, dtype=float)/10


def spatial_folds(parent_indices):
    """Five folds of 8x8-parent spatial tiles on the fixed 40x40 coarse grid.

    This layout is a project adaptation, not a rule reported in Yoo's paper.
    It does not make adjacent tiles or their temperatures independent samples.
    """
    p = np.asarray(parent_indices)
    if p.ndim != 1 or not np.issubdtype(p.dtype, np.integer):
        raise ValueError('Explicit integer parent indices required')
    if len(np.unique(p)) != len(p) or np.any((p < 0) | (p >= 1600)):
        raise ValueError('Parent identities must be unique on the 40x40 grid')
    row, col = p//40, p % 40
    return ((row//8)+2*(col//8)) % 5


def _fit(x, y, alpha):
    if np.ptp(y) == 0:
        return np.zeros(x.shape[1]), float(y[0]), 0
    # Explicitly no standardization: the historical kernels retain their K scale.
    model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=100000,
                  tol=1e-8, selection='cyclic', warm_start=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y)
    return model.coef_.copy(), float(model.intercept_), int(model.n_iter_)


def select_kernels(history_coarse_k, query_coarse_k, parent_indices, *, kernel_names):
    """Fit five-fold LASSO and retain strictly positive final coefficients.

    Unpenalized intercept; (1/2N)*SSE + alpha*L1; pooled validation MSE.
    A mathematical exact tie takes the larger penalty. No fine target is read.
    """
    x = np.asarray(history_coarse_k, float)
    y = np.asarray(query_coarse_k, float)
    names = [str(s) for s in kernel_names]
    if x.ndim != 2 or x.shape[1] == 0 or len(x) < 30:
        raise ValueError('At least 30 finite coarse parents and one kernel required')
    if y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite matched input kernels and public coarse response required')
    if len(names) != x.shape[1] or len(set(names)) != len(names):
        raise ValueError('Unique kernel names must match matrix columns')
    folds = spatial_folds(parent_indices)
    if len(folds) != len(x) or set(folds) != set(range(5)):
        raise ValueError('The fixed support must represent all five spatial folds')
    if min(int(np.sum(folds != k)) for k in range(5)) < 20:
        raise ValueError('Every spatial training fold requires at least 20 parents')
    scores = []
    maximum_iterations = 0
    with threadpool_limits(limits=1):
        for alpha in ALPHAS:
            pred = np.full(len(y), np.nan)
            for k in range(5):
                train, test = folds != k, folds == k
                coef, intercept, iterations = _fit(x[train], y[train], alpha)
                pred[test] = intercept+x[test] @ coef
                maximum_iterations = max(maximum_iterations, iterations)
            scores.append(float(np.mean((pred-y)**2)))
        best = max(i for i, value in enumerate(scores) if value == min(scores))
        alpha = ALPHAS[best]
        coef, intercept, iterations = _fit(x, y, alpha)
    selected = np.flatnonzero(coef > 0)
    return dict(
        schema='yoo-lasso-spatial-adaptation-v1',
        response='observed_current_coarse_LST_only',
        loss='SSE/(2*N) + alpha*sum(abs(coef))',
        alpha_grid=ALPHAS.tolist(), spatial_fold_id=folds.tolist(),
        fold_layout='8x8-parent tiles; (tile_row+2*tile_col) modulo 5',
        original_paper_specifies_spatial_fold_layout=False,
        standardization=False, unpenalized_intercept=True,
        coefficient_sign_constraint=False, selection_rule='fitted_coefficient_strictly_positive',
        cv_score='pooled_heldout_coarse_MSE', exact_tie_rule='larger_alpha',
        selected_alpha=float(alpha), cv_mse_k2=scores,
        coefficients=coef.tolist(), intercept_k=intercept,
        selected_indices_zero_based=selected.tolist(),
        selected_kernel_names=[names[i] for i in selected], kernel_names=names,
        kernel_count=len(names), training_parent_count=len(y),
        maximum_cv_iterations=maximum_iterations, final_iterations=iterations,
        status='selected' if len(selected) else 'no_positive_thermal_kernel',
        final_reconstruction_is_not_lasso_output=True,
    )
