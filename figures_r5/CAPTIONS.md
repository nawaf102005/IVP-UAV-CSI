## system_model

System model and frame timeline. Solid arrows connect each UAV to its BS. Dashed arrows show downlink interference under spatial reuse. Reports precede uplink data and downlink control. Decision rounds also consume time in~\eqref{eq:abstractcost}.

Source: `Original analyze_results.py schematic`

## q_accuracy_independent

Independent value audit: 3,072 candidate values in 192 states. Within-state errors are dependent. Empirical distributions; no confidence intervals.

Source: `independent_numerical_results/values.csv; summary.json; protocol.json`

## aligned_frontier_gain

Acquisition gain with certificate-based and frontier-aware forecasts and the same terminal frontier. Each cost uses 512 paired frames ($N=4$, $T=20$ ms). Bars give simultaneous 95\% intervals from the 60-contrast family. The cost axis gives the hypothetical charge per acquisition round.

Source: `dense_alignment_results/summary.json; boundary_summary.json`

## allocator_persistent_age

Radio-only C-IVP gain over matched no-report at 20 m/s on 128 independent 40-frame trajectories. Both terminal allocators use the same acquisition histories. Bars are simultaneous 95\% paired intervals from the original 192 comparisons. Tick labels give the opportunity period $K$ and the shadow correlation $r^K$.

Source: `prospective_results/reuse_contrasts.csv`

## frontier_budget_6

C-IVP and margin sensitivity against the same no-report frontier, six pairs and 50-ms frames. Markers show means over 512 independent paired frames per tested cost. Shading joins the simultaneous 95\% intervals at the eleven grid points (66 contrasts).

Source: `breakeven_results/curves.csv`

## frontier_budget_8

Gain against the same no-report frontier, eight pairs and 100-ms frames. Each cost uses 512 independent paired frames. Shading joins the simultaneous 95\% grid intervals from the same 66-contrast family as Fig.~\ref{fig:budget6}.

Source: `breakeven_results/curves.csv`

## r5_03_breakeven_B

Completed-rate gain after charging radio airtime and reserved computation. CDF denotes C-IVP. Reservations use 128 development frames and each duration uses 256 paired test frames. Bars give simultaneous 95\% intervals over fourteen contrasts with no-report. Channels are frozen and payload scales with frame duration.

Source: `round5_results/latency_records.jsonl; reservations.json`

## r5_06_raytracing_D

Outage after endpoint motion at 256 test positions per scene. Bars give pointwise 95\% Wilson intervals and the dotted line marks the 5\% target. Safe-IVP and Safe-NR use a filter calibrated on static channels. NR denotes no-report.

Source: `revision_results/ray_records.jsonl; conformal_records.jsonl; ray_data/`

## catalogue_pareto_A

Forecast-catalogue and screening tradeoff on 128 paired $N=6$ timing frames. C16/C32 retain 16/32 subsets; S2 retains two reports. Terminal admission remains exhaustive.

Source: `revision_results/summary.csv: timing rows`

## catalogue_pareto_B

Forecast-catalogue and screening tradeoff on 128 paired $N=8$ timing frames. Terminal admission remains exhaustive.

Source: `revision_results/summary.csv: timing rows`

## r5_05_risk_budget_A

Frozen design-risk sweep, 512 frames per setting. Horizontal Wilson and vertical t intervals are pointwise 95\%; the sweep does not impose equal realized risk.

Source: `revision_results/revision_records.jsonl`

## r5_05_risk_budget_B

Completed service versus report cap on 512 paired frames per setting. Policies may stop early. Bars: pointwise 95\% t intervals.

Source: `revision_results/revision_records.jsonl`

## r5_10_allocator_A

Completed service for four terminal allocators on 128 matched frames. Bars: pointwise 95\% t intervals; finite-frontier validation differs from the Gaussian certificate.

Source: `revision_results/chance_records.jsonl; frontier_records.jsonl; hybrid_records.jsonl`

## r5_07_approximation_A

Scalar reference-minus-inner shortfall on 1,024 report candidates from 128 states and 22,377 detected intervals. One interval is missed at 25 knots. The 0.28574-kbit maximum is not bounded by the 0.002-kbit stopping margin.

Source: `round2_results/interval_summary.json; round3_results/vector_bound_records.jsonl`

## r5_07_approximation_B

Mean and maximum vector value enclosure gaps on 32 two-component states. Enclosures concern the Gaussian surrogate under the stated arithmetic assumptions; no statistical confidence intervals.

Source: `round2_results/interval_summary.json; round3_results/vector_bound_records.jsonl`

## r5_07_approximation_C

Mean computation time for vector box budgets on 32 two-component states. Enclosure cost grows with the box budget.

Source: `round2_results/interval_summary.json; round3_results/vector_bound_records.jsonl`

## r5_08_search_B

Mean unresolved certified-utility gap on 32 states per cluster size and node budget. Substantial gaps remain at $N=24$.

Source: `round3_results/branch_records.jsonl`

## r5_08_search_C

Mean wall time on 32 states per cluster size and node budget. Descriptive means; no confidence intervals or 20-ms deadline guarantee.

Source: `round3_results/branch_records.jsonl`

## r5_09_pilots_A

Standardized physical-pilot log-power errors: 2,048 residuals per pilot length, clustered in 256 frames. The Gaussian curve is a model reference, not a verified physical posterior. Empirical distributions; no confidence intervals.

Source: `round4_results/pilot_observations.jsonl; pilot_value_audit.jsonl`

## r5_09_pilots_B

Absolute conditional-CDF17 versus product-GH33 discrepancy on 16 prespecified selected batches per pilot length. GH33 is a comparator, not exact physical-posterior truth. Empirical distributions; no confidence intervals.

Source: `round4_results/pilot_observations.jsonl; pilot_value_audit.jsonl`

## value_accuracy_A

Reference-minus-inner value on 1,024 report candidates from 128 held-out initial states. Positive gaps are shortfalls; negative gaps are numerical overestimates. The abscissa is linear within $10^{-6}$ kbit. Empirical distribution; no confidence intervals.

Source: `round2_results/interval_grid.csv; interval_summary.json`

## r5_01_scaling_A

Completed service on 256 paired frames per cluster size. Report airtime is charged; computation is timed separately. Bars: pointwise 95\% t intervals.

Source: `round5_results/scale_records.jsonl`

## r5_01_scaling_C

Empirical wall-time distributions at $N=8$ on 256 paired frames. Compilation and scenario generation are excluded. Distribution summaries; no confidence intervals.

Source: `round5_results/scale_records.jsonl`

## r5_02_equivalence_B

Ratio of mean KG17 to CDF runtimes on 256 paired frames per cluster size. Bars: pointwise paired-bootstrap 95\% intervals from 2,000 resamples.

Source: `R5 scale; protocol and r5_statistics.py`

## r5_03_breakeven_A

Completed rate after radio and computation charges on 256 held-out paired frames per duration. Bars: pointwise 95\% t intervals. Payload scales with duration.

Source: `round5_results/latency_records.jsonl; reservations.json`

## r5_03_breakeven_C

Mean reserved computation relative to frame duration on 256 paired test frames per duration. Reservations use 128 separate development frames; payload scales with duration.

Source: `round5_results/latency_records.jsonl; reservations.json`

## r5_04_mismatch_A

Completed service versus true UL-DL correlation, with 512 paired frames per setting. Assumed correlation is 0.9; shadowing SD is 3 dB. Bars: pointwise 95\% t intervals; matched methods know the true covariance.

Source: `round5_results/mismatch_records.jsonl`

## r5_04_mismatch_B

Completed service versus true shadowing SD, with 512 paired frames per setting. The assumed SD is 3 dB and correlation is 0.9. Bars: pointwise 95\% t intervals; matched methods know the true covariance.

Source: `round5_results/mismatch_records.jsonl`

## r5_04_mismatch_D

Any-task outage versus true shadowing SD on the same 512 paired frames per setting. Bars: pointwise 95\% Wilson intervals; the horizontal dotted line marks 5\%. Matched covariance is an oracle diagnostic.

Source: `round5_results/mismatch_records.jsonl`

## r5_06_raytracing_A

Completed service for static incoherent ray-traced gains, with 256 test frames per scene. Bars: pointwise 95\% t intervals. Safe methods use the marginal filter; these are simulations.

Source: `revision_results/ray_records.jsonl; conformal_records.jsonl; ray_data/`

## r5_06_raytracing_C

Completed service under coherent endpoint-aging stress, with 256 test frames per scene. Bars: pointwise 95\% t intervals. Changing combining and endpoint geometry does not isolate a pure motion effect.

Source: `revision_results/ray_records.jsonl; conformal_records.jsonl; ray_data/`

## value_accuracy_B

Scalar-grid sensitivity with eight refinements on 1,024 candidates from 128 states and 22,377 reference-detected intervals. Maximum/mean shortfalls are descriptive, not guaranteed approximation bounds; no confidence intervals.

Source: `round2_results/interval_grid.csv; interval_summary.json`

## r5_10_allocator_B

Paired acquisition gain with the original and stronger frontier allocators on the same 128 frames. Bars: pointwise 95\% t intervals.

Source: `revision_results/chance_records.jsonl; frontier_records.jsonl; hybrid_records.jsonl`

## r5_08_search_A

Mean incumbent-to-upper-bound ratio on 32 fixed states per cluster size and node budget. This is an instance-specific certificate, not a uniform approximation factor. Descriptive ratios; no statistical confidence intervals.

Source: `round3_results/branch_records.jsonl`

## r5_02_equivalence_A

CDF-minus-KG17 service differences on 256 paired frames per setting. Bars: pointwise 90\% paired-$t$ intervals with the prespecified $\pm0.5$-kbit region. Stars denote Holm-adjusted TOST rejection across six tests. Identical $N=6$ S2 differences are not $t$-tested.

Source: `R5 scale; protocol and r5_statistics.py`

## r5_04_mismatch_C

Any-task outage versus true UL-DL correlation, 512 paired frames per setting. Bars: pointwise 95\% Wilson intervals; the dotted line marks 5\%. Matched covariance is an oracle diagnostic.

Source: `round5_results/mismatch_records.jsonl`

## r5_06_raytracing_B

Any-task outage for static ray-traced gains, 256 test frames per scene. Bars: pointwise 95\% Wilson intervals; the dotted line marks 5\%. Scene categories are not connected.

Source: `revision_results/ray_records.jsonl; conformal_records.jsonl; ray_data/`

## cost_tolerance_margin

Sensitivity-rule empirical first-crossing budgets in the 30-cell speed/report-cost map, 128 paired frames per cell. Descriptive crossings; pointwise paired-bootstrap 95\% intervals are in Section S12. The shadow-correlation length is illustrative.

Source: `platform_cost_results/boundary_summary.json; boundaries.csv`

## cost_support_map

Largest initial tested cost prefix with positive simultaneous 95\% lower limits for IVP versus no-report, 128 paired frames per cell in the earlier 1,194-contrast family. This discrete support differs from interpolated empirical crossings.

Source: `platform_cost_results/boundary_summary.json; boundaries.csv`

## q_accuracy

Absolute report-value errors on 256 candidates in 32 states. Candidate errors within a state are dependent. The scale is linear near zero. Empirical distributions; no confidence intervals.

Source: `numerical_results/values.csv; decisions.json; protocol.json`

## mobility_radio

Radio-only predictive acquisition gain, $N=4$, $T=20$ ms, 128 paired frames per cell. A star marks a positive simultaneous 95\% lower limit in the 120-contrast family. Colors/numbers show kbit/frame.

Source: `mobility_cost_results/summary.json; protocol.json; reservations.json`

## mobility_charged

Predictive acquisition gain after method-specific computation charges, $N=4$, $T=20$ ms, 128 paired frames per cell. Simultaneous 95\% comparisons use the 120-contrast family; no cell establishes positive gain over no-report. Colors match Fig.~\ref{fig:mobilityradio}.

Source: `mobility_cost_results/summary.json; protocol.json; reservations.json`

## cost_tolerance_map

Empirical first-crossing budget $100\widehat\chi_\star$ for certificate-based predictive IVP, 128 paired frames per cell, $N=4$, $T=20$ ms, illustrative $d_{\rm cor}=10$ m. Pointwise paired-bootstrap 95\% intervals are tabulated in Section S12.

Source: `platform_cost_results/boundary_summary.json; boundaries.csv`

## margin_duration_boundaries

Sensitivity-rule first-crossing budgets, 128 paired frames per setting, with the same hypothetical per-round cost and sufficient terminal certificate. Descriptive crossings; pointwise paired-bootstrap 95\% intervals are tabulated in Section S12.

Source: `platform_cost_results/boundary_summary.json`

## size_duration_boundaries

Earlier IVP size/duration slices, 128 paired states per setting. First crossings have pointwise paired-bootstrap 95\% intervals. Fixed $v=20$ m/s, illustrative $d_{\rm cor}=10$ m, $\tau=.75$ ms, payloads proportional to $T$.

Source: `platform_cost_results/boundary_summary.json`

## frontier_cost_break_even

C-IVP minus no-report with a common terminal frontier on 512 independent paired frames per setting. Markers show tested costs. Shaded bands connect simultaneous 95\% intervals at the eleven grid points (66 contrasts). Open diamonds mark the first zero of the linearly interpolated mean.

Source: `breakeven_results/curves.csv; breakeven_results/crossings.json`

## frontier_cost_slice

Earlier predictive cohort E: acquisition gain with matched terminal allocators, $N=4$, $T=20$ ms, 128 paired frames per cost. Bars: simultaneous 95\% paired-$t$ intervals in the 1,194-contrast family. Both IVP and sensitivity retain certificate-based forecasts.

Source: `platform_cost_results/frontier_N4_T20_summary.json; slice_N4_T20_summary.json`

## dense_cost_refinement

Cohort F: 512-frame budget sweep with certificate-based forecasting and terminal allocation. Bars: simultaneous 95\% paired-$t$ intervals over 60 contrasts. Dashed: largest initial tested prefix with positive simultaneous lower limits. Dotted: first interpolated mean crossing.

Source: `dense_alignment_results/summary.json; boundary_summary.json`

## frontier_dense_replay

Completed-service gain over frontier no-report across the 17 budgets of Fig.~\ref{fig:densecost}. Both certificate-based acquisition rules use the same final frontier on the 512 archived cohort F frames. Shading gives simultaneous 95\% paired-$t$ intervals from the retrospective 58-contrast family. At $T=20$ ms, each 1\% on the horizontal axis is .20 ms per round.

Source: `final_replay_results/frontier_contrasts.csv`

## r5_01_scaling_B

Cohort H: whole-decision wall time on 256 paired frames per cluster size. Bars: pointwise 95\% $t$ intervals. Scenario generation and compilation are excluded; terminal admission remains exhaustive.

Source: `round5_results/scale_records.jsonl`

## persistent_frontier_gain

Paired service gain from optional C-IVP with a common terminal frontier and an assumed 1.72-ms charge per acquisition round. Bars give simultaneous 95\% intervals over 192 comparisons on 128 independent 40-frame trajectories per speed. Horizontal offsets separate speeds at the same $K$.

Source: `prospective_results/reuse_contrasts.csv`

## frontier_planning_resolution

Frontier-aware service gain over frontier no-report before charging computation, against the 95th percentile of whole-decision time on 128 paired frames. Lines join increasing midpoint counts (9, 25, 49) at fixed planning draws. Bars give simultaneous 95\% intervals from the 18-contrast family. The dotted vertical line marks 1 ms.

Source: `prospective_results/profile_summary.csv`
