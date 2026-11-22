"""Posterior log-ratio chance certificate and exact subset admission.

The certificate is sufficient for physical chance feasibility; it is NOT an
exact reformulation of a joint lognormal SINR chance constraint.
"""
from itertools import combinations
import numpy as np
from scipy.stats import norm
from model import thresholds, noise_power


def subsets(n):
    return [np.array(s, dtype=int) for r in range(1, n+1) for s in combinations(range(n), r)]


def coefficients(cfg, mean, cov, payload, count, group, direction=None,
                 nominal=False, diagonal=False, orthogonal=False):
    """Log-affine F(z),u(z) for a fixed subset and optional scalar innovation."""
    n, size = cfg.n, len(group)
    quantile = 0.0 if nominal else norm.isf(cfg.outage / (2 * size * size))
    theta = thresholds(cfg, payload, count, size if orthogonal else 1)[:, group]
    log_f, slope_f, log_u, slope_u = [], [], [], []
    if direction is None:
        direction = np.zeros_like(mean)
    diag_cov = np.diag(cov)
    for d in range(2):
        ix = d*n*n + group[:, None]*n + group[None, :]
        desired = d*n*n + group*(n+1)
        diff_var = diag_cov[ix] + diag_cov[desired][:, None]
        if not diagonal:
            diff_var -= 2 * cov[ix, desired[:, None]]
        np.maximum(diff_var, 0, out=diff_var)
        lf = np.log(theta[d])[:, None] + mean[ix] - mean[desired][:, None]
        lf += quantile * np.sqrt(diff_var)
        np.fill_diagonal(lf, -np.inf)
        if orthogonal:
            lf[:] = -np.inf
        log_f.append(lf)
        slope_f.append(direction[ix] - direction[desired][:, None])
        log_u.append(np.log(theta[d]*noise_power(cfg)) - mean[desired] +
                     quantile * np.sqrt(np.maximum(diag_cov[desired], 0)))
        slope_u.append(-direction[desired])
    return np.array(log_f), np.array(slope_f), np.array(log_u), np.array(slope_u)


def minimum_power(cfg, coeff, z=0.0):
    lf, sf, lu, su = coeff
    f = np.exp(np.clip(lf + sf*z, -745, 600))
    u = np.exp(np.clip(lu + su*z, -745, 600))
    s = f.shape[-1]
    if np.max(np.abs(np.linalg.eigvals(f))) >= 1 - 1e-10:
        return None
    try:
        p = np.linalg.solve(np.eye(s)[None, :, :] - f, u[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return None
    cap = np.array([cfg.ul_power_w, cfg.dl_power_w])[:, None]
    if np.any(p < 0) or np.any(p > cap * (1 + 1e-9)):
        return None
    return p


def convex_feasibility(cfg, coeff, z):
    """max_{direction,cap} log rho(F+u e_k^T/P_k), a convex function of z."""
    lf, sf, lu, su = coeff
    f = np.exp(np.clip(lf + sf*z, -745, 600))
    u = np.exp(np.clip(lu + su*z, -745, 600))
    s = f.shape[-1]
    mats = np.broadcast_to(f[:, None, :, :], (2, s, s, s)).copy()
    cap = np.array([cfg.ul_power_w, cfg.dl_power_w])
    for k in range(s):
        mats[:, k, :, k] += u / cap[:, None]
    radius = np.max(np.abs(np.linalg.eigvals(mats)))
    return np.log(max(radius, 1e-300))


def best_schedule(cfg, mean, cov, payload, count, groups=None,
                  nominal=False, diagonal=False, orthogonal=False):
    if groups is None:
        groups = subsets(cfg.n)
    values = payload.sum(axis=0)/1000
    ordered = sorted(groups, key=lambda s: -float(values[s].sum()))
    for group in ordered:
        if not orthogonal:
            coeff = coefficients(cfg, mean, cov, payload, count, group,
                                 nominal=nominal, diagonal=diagonal)
            power = minimum_power(cfg, coeff)
            if power is not None:
                return dict(group=group, power=power, value=float(values[group].sum()), mode='reuse')
        data = orthogonal_data(cfg, mean, cov, payload, count, group, nominal=nominal)
        airtime = orthogonal_airtime(data, 0)
        if np.all(airtime.sum(axis=1) <= data['duration']*(1+1e-10)):
            power = np.broadcast_to(np.array([cfg.ul_power_w,cfg.dl_power_w])[:,None],(2,len(group))).copy()
            return dict(group=group,power=power,value=float(values[group].sum()),mode='OMA',airtime=airtime)
    return dict(group=np.array([], dtype=int), power=np.zeros((2, 0)), value=0., mode='empty')


def orthogonal_data(cfg, mean, cov, payload, count, group, direction=None, nominal=False):
    n,s=cfg.n,len(group)
    ix=np.array([group*(n+1),n*n+group*(n+1)])
    k=0. if nominal else norm.isf(cfg.outage/(2*s))
    cap=np.array([cfg.ul_power_w,cfg.dl_power_w])[:,None]
    log_snr=np.log(cap/noise_power(cfg))+mean[ix]-k*np.sqrt(np.maximum(np.diag(cov)[ix],0))
    slope=np.zeros_like(log_snr) if direction is None else direction[ix]
    duration=(cfg.frame_s-count*cfg.probe_s)*np.array([cfg.ul_fraction,1-cfg.ul_fraction])
    return dict(log_snr=log_snr,slope=slope,bits=payload[:,group],duration=duration,B=cfg.bandwidth_hz)


def orthogonal_airtime(data,z):
    rates=data['B']*np.logaddexp(0,data['log_snr']+data['slope']*z)/np.log(2)
    return data['bits']/rates


def orthogonal_feasibility(data,z):
    return float(np.max(orthogonal_airtime(data,z).sum(axis=1)/data['duration']-1))


def execute(cfg, scenario, schedule, count, orthogonal=False, true_override=None):
    group, power = schedule['group'], schedule['power']
    if not len(group):
        return dict(completed_kbit=0., admitted_kbit=0., admitted_tasks=0,
                    completed_tasks=0, outage=0, active=0, radio_energy_mj=0.)
    true = scenario.get('execution_true',scenario['true']) if true_override is None else true_override
    g = np.exp(true.reshape(2, cfg.n, cfg.n)[:, group][:, :, group])
    received = g * power[:, None, :]
    desired = np.diagonal(received, axis1=1, axis2=2)
    interference = received.sum(axis=-1) - desired
    is_oma = schedule.get('mode') == 'OMA'
    if is_oma:
        interference[:] = 0
    sinr = desired / (interference + noise_power(cfg))
    if is_oma:
        actual_bits=cfg.bandwidth_hz*schedule['airtime']*np.log2(1+sinr)
        success=np.all(actual_bits >= scenario['payload'][:,group]*(1-1e-7),axis=0)
    else:
        target = thresholds(cfg, scenario['payload'], count)[:, group]
        success = np.all(sinr >= target*(1-1e-7), axis=0)
    durations = (cfg.frame_s - count*cfg.probe_s)*np.array([cfg.ul_fraction,1-cfg.ul_fraction])
    energy = float(np.sum(power * (schedule['airtime'] if is_oma else durations[:, None]))) * 1000
    return dict(completed_kbit=float(scenario['payload'][:, group][:, success].sum()/1000),
                admitted_kbit=schedule['value'], admitted_tasks=len(group),
                completed_tasks=int(success.sum()), outage=int(not success.all()),
                active=1, radio_energy_mj=energy)
