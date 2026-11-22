"""Innovation intervals and analytic Gaussian expected certified service."""
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.special import ndtr
from model import fantasy
from admission import coefficients, convex_feasibility, subsets, orthogonal_data, orthogonal_feasibility


def feasibility_interval(cfg, coeff=None, function=None):
    """Solve a one-dimensional convex sublevel set within [-z_limit,z_limit].

    Gaussian tail mass excluded is at most 2*Phi(-z_limit). Bounds are numeric
    to endpoint_tol, not symbolic exactness or interval-arithmetic certification.
    """
    left, right = -cfg.z_limit, cfg.z_limit
    fun = function if function is not None else lambda z: convex_feasibility(cfg, coeff, z)
    fl, fr = fun(left), fun(right)
    if fl <= 0 and fr <= 0:
        return left, right
    if fl <= 0:
        mid = left
    elif fr <= 0:
        mid = right
    else:
        sol = minimize_scalar(fun, bounds=(left, right), method='bounded',
                              options={'xatol':cfg.endpoint_tol, 'maxiter':60})
        mid = sol.x
        if sol.fun > -1e-9:
            return None
    if fl > 0:
        left = brentq(fun, left, mid, xtol=cfg.endpoint_tol)
    if fr > 0:
        right = brentq(fun, mid, right, xtol=cfg.endpoint_tol)
    return float(left), float(right)


def expected_envelope(intervals):
    """Integral of max(value of feasible subset) against a standard Gaussian."""
    if not intervals:
        return 0.
    edges = sorted(set(x for lo, hi, value in intervals for x in (lo, hi)))
    result = 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = (lo+hi)/2
        value = max((v for a,b,v in intervals if a <= mid <= b), default=0.)
        result += value * (ndtr(hi)-ndtr(lo))
    return float(result)


def probe_value(cfg, mean, cov, payload, count, index, groups=None,
                diagonal=False, return_intervals=False):
    if groups is None:
        groups = subsets(cfg.n)
    noise_var = (cfg.measurement_db*np.log(10)/10)**2
    direction, new_cov = fantasy(cov, index, noise_var)
    intervals = []
    weights = payload.sum(axis=0)/1000
    for group in groups:
        coeff = coefficients(cfg, mean, new_cov, payload, count+1, group,
                             direction=direction, diagonal=diagonal)
        bounds = feasibility_interval(cfg, coeff)
        if bounds:
            intervals.append((*bounds, float(weights[group].sum())))
        if len(group)>1:
            data=orthogonal_data(cfg, mean, new_cov, payload, count+1, group, direction=direction)
            bounds=feasibility_interval(cfg,function=lambda z:orthogonal_feasibility(data,z))
            if bounds:
                intervals.append((*bounds,float(weights[group].sum())))
    value = expected_envelope(intervals)
    return (value, intervals) if return_intervals else value
