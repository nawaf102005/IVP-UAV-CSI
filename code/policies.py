"""All causal methods share the report catalogue, hardware caps and samples."""
import time
import numpy as np
from model import posterior, fantasy
from admission import best_schedule, execute, subsets
from innovation import probe_value


METHODS = ['IVP', 'A-opt-stop', 'D-opt-stop', 'Random-fixed', 'No-report',
           'Orthogonal', 'Nominal', 'Perfect-CSI']


def run_policy(cfg, sc, method, fixed_probes=2, interval_log=False):
    started = time.perf_counter()
    mean, cov = sc['mean'].copy(), sc['covariance'].copy()
    groups = subsets(cfg.n)
    noise_var = (cfg.measurement_db*np.log(10)/10)**2
    nreport = 0
    observations = []
    use_count = np.zeros(len(mean), dtype=int)
    rng = np.random.default_rng(sc['seed']+899999)
    if method == 'Perfect-CSI':
        mean = sc.get('execution_true',sc['true']).copy()
        cov[:] = 0
    diagonal = method == 'IVP-diagonal'
    if method in ['IVP','IVP-diagonal','A-opt-stop','D-opt-stop','Random-fixed','A-opt-fixed']:
        for m in range(cfg.max_probes):
            current = best_schedule(cfg, mean, cov, sc['payload'], nreport, groups,
                                    diagonal=diagonal)
            if current['value'] >= sc['payload'].sum()/1000 - 1e-9:
                break
            candidates = sc['candidates']
            if method in ['IVP','IVP-diagonal']:
                estimates = [probe_value(cfg, mean, cov, sc['payload'], nreport,
                                         int(k), groups, diagonal=diagonal) for k in candidates]
                index = int(candidates[int(np.argmax(estimates))])
                estimate = max(estimates)
                if estimate <= current['value'] + cfg.gain_tol_kbit:
                    break
            elif method in ['A-opt-stop','A-opt-fixed','D-opt-stop']:
                if method.startswith('A-opt'):
                    scores = np.sum(cov[:, candidates]**2, axis=0)/(np.diag(cov)[candidates]+noise_var)
                else:
                    scores = np.log1p(np.diag(cov)[candidates]/noise_var)
                index = int(candidates[int(np.argmax(scores))])
                if method.endswith('stop'):
                    estimate = probe_value(cfg, mean, cov, sc['payload'], nreport, index, groups)
                    if estimate <= current['value'] + cfg.gain_tol_kbit:
                        break
                elif nreport >= fixed_probes:
                    break
            else:
                if nreport >= fixed_probes:
                    break
                index = int(rng.choice(candidates))
            obs = sc['true'][index] + np.sqrt(noise_var)*sc['report_noise'][index, use_count[index]]
            use_count[index] += 1
            mean, cov = posterior(mean, cov, index, obs, noise_var)
            nreport += 1
            observations.append(dict(index=index, observation=float(obs)))
    nominal = method in ['Nominal', 'Perfect-CSI']
    schedule = best_schedule(cfg, mean, cov, sc['payload'], nreport, groups,
                             nominal=nominal, diagonal=diagonal, orthogonal=method=='Orthogonal')
    elapsed = time.perf_counter()-started
    result = execute(cfg, sc, schedule, nreport, orthogonal=method=='Orthogonal')
    result.update(method=method, seed=sc['seed'], n=cfg.n, probes=nreport,
                  report_fraction=nreport*cfg.probe_s/cfg.frame_s,
                  wall_s=elapsed, selected=list(map(int,schedule['group'])),
                  mode=schedule['mode'], power=schedule['power'].tolist(), observations=observations)
    if 'airtime' in schedule:
        result['airtime']=schedule['airtime'].tolist()
    return result
