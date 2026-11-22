"""Presentation-only aggregation of frozen evidence; no policy run or new test sample."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def save(path,lines):path.write_text('\n'.join(lines)+'\n')
def main(root):
    root=Path(root);t=root/'manuscript';q=json.loads((root/'independent_numerical_results/summary.json').read_text())['All'];work=json.loads((root/'certificate_work_results/summary.json').read_text());grid=json.loads((root/'round2_results/interval_summary.json').read_text())['results']
    # reference_wall_s is repeated once per comparator; count each stored root
    # evaluation once. The independent instrumented replay supplies separate work counts.
    root_times={}
    with (root/'independent_numerical_results/values.csv').open() as f:
        for row in csv.DictReader(f):
            key=(row['case'],row['seed'],row['index']);value=float(row['reference_wall_s'])
            if key in root_times:assert root_times[key]==value
            root_times[key]=value
    assert len(root_times)==3072
    root_ms=1000*np.mean(list(root_times.values()))
    from render_final_replay import VALUE_CAPTION, VALUE_NOTE
    rc=json.loads((root/'final_replay_results/root_summary.json').read_text())
    lines=[r'\begin{table*}[t]\centering\small',r'\caption{'+VALUE_CAPTION+r'}\label{tab:valuemain}',r'\begin{tabular}{@{}lrrrrrrr@{}}\toprule',r'Method & Mean $|\Delta Q|$ & Max $|\Delta Q|$ & Decisions & Feas. evals & Lin. solves & Prep. & Mean ms\\\midrule',f'Root reference & Ref. & Ref. & Ref. & {rc["evaluations"]:.1f} & {rc["linear_solve_calls"]:.0f} & {rc["reuse_preparations"]:.0f}+{rc["oma_preparations"]:.0f} & {root_ms:.3f}'+r'\\']
    for method in ['CDF25','KG17','Adaptive-GK']:
        a=q[method];b=work[method];lines.append(f'{method} & {a["mean_abs_error"]:.3g} & {a["max_abs_error"]:.3g} & {a["decision_agreement"]}/192 & {b["certificate_tests"]["mean"]:.1f} & {b["directional_linear_solves"]["mean"]:.1f} & {b["action_preparations"]["mean"]:.1f} & {a["mean_ms"]:.3f}'+r'\\')
    lines+=[r'\bottomrule\end{tabular}',VALUE_NOTE];save(t/'evidence_value_table.tex',lines)
    lines=[r'\begin{table}[t]\centering\small',r'\caption{Grid sensitivity on the earlier 128 states, 1,024 candidates, and 22,377 reference-detected nonempty intervals. Eight refinements; gaps are $Q_{\rm ref}-Q_G$ in kbit. Descriptive errors/counts; no confidence interval.}\label{tab:gridmain}',r'\begin{tabular}{@{}rrrr@{}}\toprule',r'Knots & Missed intervals & Mean gap & Max gap\\\midrule']
    for d in grid:lines.append(f'{d["grid"]} & {d["missed"]} & {d["mean_gap"]:.5f} & {d["max_gap"]:.5f}'+r'\\')
    lines+=[r'\bottomrule\end{tabular}\end{table}'];save(t/'evidence_grid_table.tex',lines)
    lines=[r'\begin{table}[t]\centering\small',r'\caption{Original 1,024-frame service block. Radio report airtime is charged; computation is not deducted. Outage means any admitted-task failure. Descriptive means/counts; no confidence interval.}\label{tab:servicemain}',r'\begin{tabular}{@{}lrrr@{}}\toprule',r'Rule & kbit/frame & Reports & Outages\\\midrule',r'C-IVP & 106.93 & 1.15 & 19/1024\\',r'A-stop & 99.81 & .50 & 22/1024\\',r'KG17 & 106.88 & 1.14 & 19/1024\\',r'No report & 98.06 & 0 & 17/1024\\',r'\bottomrule\end{tabular}\end{table}'];save(t/'evidence_service_table.tex',lines)
    latency=[json.loads(l) for l in (root/'round5_results/latency_records.jsonl').read_text().splitlines()];timing=json.loads((root/'dense_alignment_results/timing_summary.json').read_text())
    lines=[r'\begin{table*}[t]\centering\small',r'\caption{Two distinct timing studies. (a) A causal reservation reduces delivery time on 256 static 20-ms frames; active overruns are penalized. (b) Dedicated timing on 32 execution-target frames; no service-after-latency claim is attached to these means. Descriptive means/counts/percentiles; no confidence interval.}\label{tab:timingmain}',r'\begin{minipage}[t]{.49\textwidth}\centering',r'(a) Reserved-time service, 256 frames\par\smallskip',r'\begin{tabular}{@{}lrrrr@{}}\toprule',r'Rule & $c$ ms & Wall ms & kbit & Outages\\\midrule']
    for method,label in [('CDF','C-IVP'),('Margin','Sensitivity'),('NR','No report')]:
        rs=[r for r in latency if r['duration_ms']==20 and r['method']==method];assert len(rs)==256;lines.append(f'{label} & {1000*rs[0]["reserve_s"]:.2f} & {1000*np.mean([r["wall_s"] for r in rs]):.2f} & {np.mean([r["completed_kbit"] for r in rs]):.2f} & {sum(r["outage"] for r in rs)}/256'+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{minipage}\hfill',r'\begin{minipage}[t]{.49\textwidth}\centering',r'(b) Timing only, 32 frames\par\smallskip',r'\begin{tabular}{@{}lrrr@{}}\toprule',r'Rule & Mean ms & 95th pct. & Rounds\\\midrule']
    for method,label in [('Certificate-IVP','C-IVP'),('Frontier-aware-IVP','Frontier-aware'),('No-report-C','No-report, cert.'),('No-report-F','No-report, frontier')]:
        d=timing[method];lines.append(f'{label} & {d["mean_decision_ms"]:.2f} & {d["p95_decision_ms"]:.2f} & {d["mean_rounds"]:.3f}'+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{minipage}\end{table*}'];save(t/'evidence_timing_table.tex',lines)
    lines=[r'\begin{table*}[t]\centering\small',r'\caption{Matched acquisition gains under different allocators. No-report and reporting service are kbit/frame. P: pointwise 95\% interval in the frozen 128-frame cohort; S: simultaneous 95\% interval over the fresh 512-frame, 60-contrast family. Rows from different cohorts are not pooled.}\label{tab:allocmain}',r'\begin{tabular}{@{}llrrlr@{}}\toprule',r'Cohort & Forecast / final allocator & No report & Reporting & Paired acquisition gain & Outages NR / report\\\midrule',r'128 frozen & Certificate / certificate & 96.88 & 108.80 & $11.92\ [9.27,14.58]^{\rm P}$ & 4 / 3\\',r'128 frozen & Certificate / frontier & 115.28 & 117.21 & $1.93\ [.66,3.20]^{\rm P}$ & 2 / 0\\',r'512 predictive & Certificate / frontier & 115.80 & 116.44 & $.649\ [-.293,1.590]^{\rm S}$ & 10 / 9\\',r'512 predictive & Frontier / frontier & 115.80 & 119.07 & $3.279\ [1.907,4.651]^{\rm S}$ & 10 / 12\\',r'\bottomrule\end{tabular}\end{table*}'];save(t/'evidence_allocator_table.tex',lines)
    lines=[r'\begin{table*}[t]\centering\small',r'\caption{Observation and execution-channel diagnostics on separate 256-frame blocks. All service values deduct radio overhead only. Pilot timings exclude waveform generation and compilation. Descriptive means/counts; no confidence interval.}\label{tab:observationmain}',r'\begin{minipage}[t]{.49\textwidth}\centering',r'(a) Orthogonal two-pilot observation simulation\par\smallskip',r'\begin{tabular}{@{}rlrrr@{}}\toprule',r'$L_{\rm p}$ & Rule & kbit & Outages & Wall ms\\\midrule',r'16 & Cond.-CDF & 108.41 & 2/256 & 17.07\\',r'16 & No report & 98.73 & 2/256 & .09\\',r'64 & Cond.-CDF & 108.90 & 2/256 & 16.34\\',r'64 & No report & 98.73 & 2/256 & .09\\',r'\bottomrule\end{tabular}\end{minipage}\hfill',r'\begin{minipage}[t]{.49\textwidth}\centering',r'(b) Separately calibrated canyon execution endpoint\par\smallskip',r'\begin{tabular}{@{}lrr@{}}\toprule',r'Rule & kbit & Outages\\\midrule',r'Frozen-Safe & 233.32 & 56/256\\',r'Predictive-IVP & 350.80 & 15/256\\',r'Endpoint-Safe & 243.83 & 0/256\\',r'Endpoint-NR & 243.98 & 0/256\\',r'\bottomrule\end{tabular}\end{minipage}\end{table*}'];save(t/'evidence_observation_table.tex',lines)
    print('Built six evidence tables from frozen records and previously verified summaries')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();main(a.root)
