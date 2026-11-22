"""Synthetic, reciprocal lognormal UAV-corridor channel and CSI reports.

No small-scale fading, hardware measurements, or learned priors are claimed.
Indices: g[d, i, j] is the power gain from transmitter j to receiver i;
d=0 denotes uplink and d=1 downlink. All logs are natural logarithms.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Config:
    n: int = 4
    frame_s: float = 0.020
    bandwidth_hz: float = 1e6
    ul_fraction: float = 0.8
    probe_s: float = 0.00075
    max_probes: int = 4
    ul_bits: float = 30000.0
    dl_bits: float = 1800.0
    ul_power_w: float = 0.2
    dl_power_w: float = 1.0
    noise_figure_db: float = 7.0
    shadow_db: float = 3.0
    measurement_db: float = 1.0
    reciprocity: float = 0.90
    spacing_m: float = 200.0
    altitude_m: float = 120.0
    kappa: float = 4.0
    pathloss_exponent: float = 2.7
    excess_loss_db: float = 15.0
    outage: float = 0.05
    z_limit: float = 6.0
    endpoint_tol: float = 2e-5
    gain_tol_kbit: float = 0.002


def scenario(cfg: Config, seed: int):
    """An independent corridor snapshot, including pre-generated common noise."""
    rng = np.random.default_rng(seed)
    n = cfg.n
    columns = (n + 1) // 2
    base = np.array([[i % columns, i // columns, 0.] for i in range(n)])
    base[:, :2] *= cfg.spacing_m
    base[:, :2] += rng.uniform(-10, 10, (n, 2))
    uav = base.copy()
    uav[:, 0] += rng.uniform(-40, 40, n)
    uav[:, 1] += rng.uniform(-10, 10, n)
    uav[:, 2] = cfg.altitude_m + 30 * (np.arange(n) % 2)
    # Reciprocal BS beams point to the paired UAV. UAV antennas are isotropic.
    boresight = uav - base
    boresight /= np.linalg.norm(boresight, axis=1)[:, None]
    vec = uav[None, :, :] - base[:, None, :]
    dist = np.linalg.norm(vec, axis=2)
    cosine = np.einsum('ij,ikj->ik', boresight, vec / dist[:, :, None])
    gain = 2 * (cfg.kappa + 1) * np.maximum(cosine, 0) ** cfg.kappa + 0.02
    beta = (299792458 / (4 * np.pi * 3.5e9)) ** 2
    ul = beta * gain / dist ** cfg.pathloss_exponent * 10 ** (-cfg.excess_loss_db / 10)
    mean = np.stack((np.log(ul), np.log(ul.T))).reshape(-1)
    # Correlated endpoint shadowing plus an independent physical-edge term.
    row, col = np.indices((n, n))
    row, col = row.ravel(), col.ravel()
    cov_base = (.4 * (row[:, None] == row[None, :]) +
                .3 * (col[:, None] == col[None, :]) + .3 * np.eye(n*n))
    cov_base *= (cfg.shadow_db * np.log(10) / 10) ** 2
    perm = np.arange(n*n).reshape(n, n).T.ravel()
    cov = np.block([[cov_base, cfg.reciprocity * cov_base[:, perm]],
                    [cfg.reciprocity * cov_base[perm, :], cov_base[np.ix_(perm, perm)]]])
    true = mean + np.linalg.cholesky(cov + 1e-14 * np.eye(2*n*n)) @ rng.normal(size=2*n*n)
    payload = np.stack((cfg.ul_bits * rng.uniform(.85, 1.15, n),
                        cfg.dl_bits * rng.uniform(.85, 1.15, n)))
    # Fixed report catalogue: direct links and strongest cross link into each BS.
    # A report is one scalar uplink gain; reciprocity informs downlink gains.
    cross = ul.copy()
    np.fill_diagonal(cross, -1)
    candidates = np.unique(np.r_[np.arange(n)*(n+1), np.arange(n)*n+cross.argmax(axis=1)])
    eps = rng.normal(size=(len(mean), cfg.max_probes+3))
    return dict(mean=mean, covariance=cov, true=true, payload=payload,
                candidates=candidates, report_noise=eps, base=base, uav=uav,
                seed=seed, config=asdict(cfg))


def posterior(mean, cov, index, observation, noise_var):
    column = cov[:, index].copy()
    variance = cov[index, index] + noise_var
    new_mean = mean + column * ((observation - mean[index]) / variance)
    new_cov = cov - np.outer(column, column) / variance
    return new_mean, (new_cov + new_cov.T) / 2


def fantasy(cov, index, noise_var):
    """mu(Z)=mu+direction*Z, Z~N(0,1); covariance does not depend on Z."""
    column = cov[:, index].copy()
    variance = cov[index, index] + noise_var
    new_cov = cov - np.outer(column, column) / variance
    return column / np.sqrt(variance), (new_cov + new_cov.T) / 2


def noise_power(cfg):
    return 10 ** ((-174 + cfg.noise_figure_db - 30) / 10) * cfg.bandwidth_hz


def thresholds(cfg, payload, count, orthogonal_size=1):
    available = cfg.frame_s - count * cfg.probe_s
    if available <= 0:
        return np.full_like(payload, np.inf)
    durations = np.array([cfg.ul_fraction, 1-cfg.ul_fraction]) * available
    return np.expm1(np.log(2) * payload * orthogonal_size /
                    (cfg.bandwidth_hz * durations[:, None]))
