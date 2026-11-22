"""Draw the acquisition-cost comparisons and tabulate the matched budget grids."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieee_figures import configure_ieee_style, style_axis
from plot_platform_revision import export
from render_prospective import maketable

AGE_TITLE = "Service gain by terminal allocator and interval between report opportunities"
COST_TITLE = "Per-round decision budget with a matched terminal frontier"
METHODS = [("C-IVP", "No-report"), ("Sensitivity", "No-report"), ("C-IVP", "Sensitivity")]


def records(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def fmt(value, precision=3):
    x = float(value)
    if 0 < abs(x) < .5 * 10 ** -precision:
        return f"{x:.2g}"
    return f"{x:.{precision}f}"


def estimate(row):
    return f"${fmt(row['mean'])}\\;[{fmt(row['lo'])},{fmt(row['hi'])}]$"


def supported(row):
    if row["block"] == "S18":
        return "Descriptive"
    if row["supported_ms"] is None:
        return "None"
    if row["supported_ms"] == 0:
        return "Zero only"
    return fmt(row["supported_ms"])


def age_chart(root):
    source = "prospective_results/reuse_contrasts.csv"
    rows = records(root / source)
    fig, ax = plt.subplots(figsize=(3.5, 2.72))
    fig.subplots_adjust(left=.165, right=.97, bottom=.235, top=.835)
    for i, (allocator, name, mark, color) in enumerate([
        ("cert", "Certificate", "o", "#145da0"), ("front", "Finite frontier", "s", "#a04a00")
    ]):
        selected = sorted((r for r in rows if r["terminal"] == allocator and int(r["speed"]) == 20
                           and float(r["cost_ms"]) == 0 and r["left"] == "IVP" and r["right"] == "No-report"),
                          key=lambda r: int(r["K"]))
        assert [int(r["K"]) for r in selected] == [1, 5, 10, 20]
        y, lo, hi = np.array([[float(r[key]) for r in selected] for key in ["mean", "lo", "hi"]])
        ax.errorbar(np.arange(4)+(i-.5)*.14, y, yerr=(y-lo, hi-y), fmt=mark,
                    color=color, markerfacecolor="white", lw=1., ms=4.8, capsize=2.6, label=name)
    ax.axhline(0, lw=.8, ls=":", color="#333333")
    ticks = [f"{k}\n({np.exp(-.04*k):.2f})" for k in (1, 5, 10, 20)]
    ax.set(xlim=(-.35, 3.35), xticks=range(4), xticklabels=ticks,
           xlabel="Opportunity period K (correlation at K)", ylabel="Service gain (kbit/frame)")
    style_axis(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
               handletextpad=.18, columnspacing=.9, bbox_to_anchor=(.54, 1.0))
    return export(fig, root, "allocator_persistent_age", AGE_TITLE,
                  "Paired radio-only gains at 20 m/s with simultaneous 95% intervals on 128 trajectories.",
                  source, 2.72)


def budget_chart(root, rows, crossings):
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    fig.subplots_adjust(left=.18, right=.965, bottom=.205, top=.83)
    for n, marker, color, linestyle in [(6, "o", "#145da0", "-"), (8, "s", "#9b490b", "--")]:
        subset = sorted((r for r in rows if r["block"] == "confirmation" and int(r["n"]) == n
                        and (r["left"], r["right"]) == METHODS[0]), key=lambda r: int(r["index"]))
        x = np.array([100*float(r["alpha"]) for r in subset])
        y, lo, hi = np.array([[float(r[key]) for r in subset] for key in ("mean", "lo", "hi")])
        spec = next(b for b in crossings if b["block"] == "confirmation" and b["n"] == n and b["rule"] == "C-IVP")
        ax.fill_between(x, lo, hi, color=color, alpha=.14, linewidth=0)
        ax.plot(x, y, marker=marker, ms=3.8, lw=1.2, linestyle=linestyle, color=color,
                markerfacecolor="white", label=f"{n} pairs, {spec['T_ms']} ms")
        ax.plot(100*spec["mean_cross_ms"]/spec["T_ms"], 0, "D", color=color, ms=5.6,
                markeredgewidth=1.25, markerfacecolor="white", zorder=6)
    ax.axhline(0, lw=.8, ls=":", color="#222222")
    ax.set(xlim=(-.055,2.555), xticks=np.arange(0,2.6,.5),
           xlabel="Per-round decision cost c/T (%)", ylabel="Gain over no-report (kbit/frame)")
    style_axis(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", frameon=False, ncol=2,
               handlelength=1.4, handletextpad=.35, columnspacing=.72, bbox_to_anchor=(.545, 1.0))
    return export(fig, root, "frontier_cost_break_even", COST_TITLE,
                  "512 independent frames per cell. Simultaneous 95% grid intervals, linear mean crossings.",
                  "breakeven_results/curves.csv; breakeven_results/crossings.json", 2.75)


def grid_tables(tex, rows):
    by = {(r['block'], int(r['n']), int(r['index']), r['left'], r['right']):r for r in rows}
    def at(block, n, i, pair):
        return by[block, n, i, *pair]
    bodies = {}
    for n in (6, 8):
        fresh = []
        raw = []
        for block, group in [("S18", "Original"), ("confirmation", "New")]:
            for i in range(11):
                ivp, margin, direct = [at(block,n,i,comp) for comp in METHODS]
                if block=="confirmation":
                    fresh.append(" & ".join([f"{i*.25:.2f}", fmt(ivp['cost_ms']), *[estimate(r) for r in (ivp,margin,direct)]]))
                def values(r):
                    return f"{fmt(r['left_service'],2)} / {r['left_outages']} / {r['left_reports']}"
                nr = f"{fmt(ivp['right_service'],2)} / {ivp['right_outages']} / 0"
                raw.append(" & ".join([group, f"{i*.25:.2f}",nr,values(ivp),values(margin)]))
        maketable(tex/f"be_confirmation_{n}.tex",f"tab:beconfirm{n}",
                  f"Independent {n}-pair cost grid, 512 paired frames. Gains in kbit/frame with simultaneous 95"+r"\% intervals for 66 contrasts. NR denotes no-report.",
                  "rrccc",r"$c/T$ (\%) & $c$ (ms) & C-IVP $-$ NR & Margin $-$ NR & C-IVP $-$ margin",fresh)
        maketable(tex/f"be_raw_{n}.tex",f"tab:beraw{n}",
                  f"Completed kbit/frame, total outages and total reports for {n} pairs, in that order. The original and independent blocks each contain 512 frames. Entries are descriptive.",
                  "lrccc",r"Frames & $c/T$ (\%) & No-report & C-IVP & Margin sensitivity",raw)
    archive = []
    for i in range(11):
        vals = [fmt(at("S18",n,i,pair)['mean']) for n in (6,8) for pair in METHODS]
        archive.append(" & ".join([f"{i*.25:.2f}",*vals]))
    maketable(tex/"be_archive.tex","tab:bearchive",
              "Original S18 grid expanded on the same 512 frames per setting. Each trio gives mean differences in kbit/frame in the order C-IVP minus no-report, margin minus no-report and C-IVP minus margin. These interpolating data are descriptive. Original endpoint intervals remain in Section S18.",
              "r"+"r"*6,r"$c/T$ (\%) & \multicolumn{3}{c}{Six pairs, 50 ms} & \multicolumn{3}{c}{Eight pairs, 100 ms}",archive)


def boundaries(tex, crossings):
    body = []
    for b in crossings:
        lo, hi = b['bracket_ms']
        ci = "---" if b['bootstrap_lo'] is None else f"$[{fmt(b['bootstrap_lo'])},{fmt(b['bootstrap_hi'])}]$"
        flags = b["bootstrap_status"]
        clipped = "---" if not flags else f"{flags.get('no_initial_gain',0)} / {flags.get('right_censored',0)}"
        body.append(" & ".join(["S18" if b["block"]=="S18" else "Independent",str(b["n"]),b["rule"],
                      f"$[{fmt(lo)},{fmt(hi)}]$",fmt(b['mean_cross_ms']),ci,supported(b),clipped]))
    maketable(tex/"be_boundaries.tex","tab:bebounds",
           r"First zero of the grid-linear mean gain against no-report (ms per round), 512 paired frames per size and block. The new samples give pointwise 95\% paired frame-bootstrap intervals. Support uses the 66-contrast simultaneous grid. The last column counts bootstrap curves with no initial gain or right censoring, respectively, out of 5,000.",
           "lrlrcccr",r"Frames & $N$ & Policy & Tested bracket & Crossing & Pointwise interval & Support & Censored",body)


def policy_budget_chart(root, grid, pairs):
    """Compare both rules against the same no-report outcome in one setting."""
    fig, ax = plt.subplots(figsize=(3.5, 2.62))
    fig.subplots_adjust(left=.18, right=.97, bottom=.215, top=.84)
    styles = [
        ("C-IVP", "C-IVP", "#155787", "o", "-"),
        ("Sensitivity", "Margin sensitivity", "#9d4b12", "s", "--"),
    ]
    for rule, label, color, marker, linestyle in styles:
        selected = sorted(
            (r for r in grid if r["block"] == "confirmation" and int(r["n"]) == pairs
             and r["left"] == rule and r["right"] == "No-report"),
            key=lambda r: int(r["index"]),
        )
        assert [int(r["index"]) for r in selected] == list(range(11))
        x = [100 * float(r["alpha"]) for r in selected]
        y, low, high = ([float(r[k]) for r in selected] for k in ("mean", "lo", "hi"))
        ax.fill_between(x, low, high, color=color, alpha=.15, linewidth=0)
        ax.plot(x, y, marker=marker, ms=3.6, lw=1.3, linestyle=linestyle,
                markerfacecolor="white", color=color, label=label)
    ax.axhline(0, color="#444444", linestyle=":", linewidth=.9)
    ax.set(xlim=(-.055, 2.555), xticks=np.arange(0, 2.6, .5),
           xlabel="Decision cost per round, c/T (%)", ylabel="Gain over no-report (kbit/frame)")
    style_axis(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, handlelength=1.6, handletextpad=.35, columnspacing=1.1,
               bbox_to_anchor=(.54, .99))
    duration = int(selected[0]["T_ms"])
    return export(fig, root, f"frontier_budget_{pairs}",
                  f"C-IVP and margin sensitivity ({pairs} pairs, {duration} ms)",
                  "Gains over the common terminal frontier on 512 independent paired frames. "
                  "Bands join simultaneous 95% intervals from 66 paired contrasts.",
                  "breakeven_results/curves.csv", 2.62)


def policy_budget_table(tex, grid, crossings, work):
    """Pair gain, crossing, statistical support and work without mixing denominators."""
    radio = {(int(r["n"]), r["left"]): r for r in grid
             if r["block"] == "confirmation" and r["index"] == "0" and r["right"] == "No-report"}
    tests = {(int(r["n"]), r["rule"]): float(r["certificate_tests_per_round"]) for r in work
             if r["block"] == "confirmation" and r["index"] == "0"}
    limits = {(r["n"], r["rule"], r["block"]): r for r in crossings}
    lines = [
        r"\begin{table*}[!t]\centering\small",
        r"\caption{C-IVP and margin sensitivity against the same terminal frontier. Each setting uses 512 original and 512 independent paired frames. Independent-block gain intervals and supported costs use the simultaneous 95\% family of 66 grid contrasts. Interpolated mean crossings have pointwise 95\% paired-bootstrap intervals on the independent block. Work is the number of certificate tests per acquisition round at zero decision cost on that block.}\label{tab:designbreak}",
        r"\begin{tabular}{@{}clccccc@{}}\toprule",
        r"& & Independent gain at $c=0$ & \multicolumn{2}{c}{Interpolated mean crossing (ms/round)} & Supported & Tests/round\\",
        r"\cmidrule(lr){4-5}",
        r"$N$ / $T$ (ms) & Reporting rule & kbit/frame [interval] & Original & Independent [interval] & grid & at $c=0$\\\midrule",
    ]
    for n in (6, 8):
        for rule, display in (("C-IVP", "C-IVP"), ("Sensitivity", "Margin sensitivity")):
            r = radio[n, rule]
            new = limits[n, rule, "confirmation"]
            old = limits[n, rule, "S18"]
            if n == 8 and rule == "C-IVP":
                lines.append(r"\addlinespace[2pt]")
            setting = f"{n} / {new['T_ms']}" if rule == "C-IVP" else ""
            gain = "$" + f"{float(r['mean']):.2f}\\;[{float(r['lo']):.2f},{float(r['hi']):.2f}]$"
            interval = "$" + f"{new['mean_cross_ms']:.3f}\\;[{new['bootstrap_lo']:.3f},{new['bootstrap_hi']:.3f}]$"
            lines.append(" & ".join([setting, display, gain, f"{old['mean_cross_ms']:.3f}",
                                     interval, supported(new), f"{tests[n, rule]:,.0f}"]) + r"\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table*}"])
    (tex / "be_main_table.tex").write_text("\n".join(lines) + "\n")


def workload(tex, work, times):
    idx = {(int(r['n']),int(r['index']),r['rule']):r for r in work if r['block']=="confirmation"}
    worktable=[]
    for i in range(11):
        cells=[]
        for n in (6,8):
            for m in ("C-IVP","Sensitivity"):
                r=idx[n,i,m]
                cells.append(f"{float(r['certificate_tests_per_round']):,.0f}")
        worktable.append(" & ".join([f"{i*.25:.2f}",*cells]))
    maketable(tex/"be_work.tex","tab:bework",r"Compiled certificate tests per acquisition round across 512 independent frames per cell, including stopping. Descriptive ratios of totals. No-report bypasses acquisition.",
              "rrrrr",r"$c/T$ (\%) & Six pairs, C-IVP & Six pairs, margin & Eight pairs, C-IVP & Eight pairs, margin",worktable)
    timing=[]
    for r in times:
        tag="Mean crossing" if r["target"]=="mean_crossing" else r"Fixed 0.5\%" if r["target"]=="fixed_half_percent" else "Supported grid"
        timing.append(" & ".join([r["n"],tag,fmt(r["budget_ms"]),r["total_rounds_per_pass"],
                      fmt(r["acquisition_ms_per_round"],2),f"{float(r['certificate_calls_per_round']):,.0f}",
                      fmt(r["required_million_calls_s"],1),fmt(r["mean_speedup_to_target"],1)+r"$\times$"]))
    maketable(tex/"be_timing.tex","tab:betiming",r"Incremental acquisition time and work for C-IVP on the same 128 confirmation identities per size. Two timed repetitions and a separate workload pass. Hardware means and target throughput are descriptive. M/s denotes millions of equivalent certificate tests per second of the whole acquisition path.",
              "rlrrrrrr",r"$N$ & Target & $c$ (ms) & Rounds/pass & ms/round & Tests/round & M/s needed & Speedup",timing)


def generate(root):
    root=Path(root); tex=root/"manuscript"; data=root/"breakeven_results"
    grid=records(data/"curves.csv"); time=records(data/"timing_summary.csv")
    crossing=json.loads((data/"crossings.json").read_text())
    grid_tables(tex,grid)
    work = records(data/"work.csv")
    boundaries(tex, crossing)
    policy_budget_table(tex, grid, crossing, work)
    workload(tex, work, time)
    for src in ("evidence_grid_table","evidence_service_table","evidence_allocator_table","evidence_timing_table",
                "evidence_tolerance_table","evidence_size_table","evidence_observation_table","ray_main_table"):
        target=tex/(src+".tex")
        if target.exists():
            s=target.read_text().replace(r"\begin{table*}[t]",r"\begin{table}[H]").replace(
                r"\end{table*}",r"\end{table}").replace(r"\begin{table}[t]",r"\begin{table}[H]")
            target.write_text(s)
    configure_ieee_style()
    return [age_chart(root), budget_chart(root, grid, crossing),
            policy_budget_chart(root, grid, 6), policy_budget_chart(root, grid, 8)]


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    print(json.dumps(generate(arg.parse_args().root),indent=2))
