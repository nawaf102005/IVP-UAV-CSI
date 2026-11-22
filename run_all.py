"""Run checks or reproduce the revised study without changing frozen archives."""
import argparse,os,shutil,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['verify','smoke','paper','round2','round3','round4','round5','figures','numerical','mobility','independent','platform','alignment','work','equations','record-replay','persistent','size-confirmation','break-even'],default='verify')
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    root=Path(__file__).resolve().parent;code=root/'code';env=os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
    def run(name,*args):subprocess.run([sys.executable,str(code/name),*map(str,args)],cwd=root,env=env,check=True)
    if a.mode=='break-even':
        output = root / 'reproduced_break_even'
        for stage in ('freeze', 'simulate', 'analyze', 'timing'):
            run('frontier_break_even.py', '--stage', stage, '--out', output, '--workers', a.workers)
        run('audit_break_even.py', '--out', output)
        return
    if a.mode=='persistent':
        output=root/'reproduced_persistent'
        for stage in ('freeze','reuse','host'):
            run('persistence_campaign.py','--stage',stage,'--out',output,'--workers',a.workers)
        run('frontier_tradeoff_campaign.py','--out',output)
        run('audit_prospective.py','--out',output);return
    if a.mode=='size-confirmation':
        output=root/'reproduced_size_confirmation'
        run('size_confirmation.py','--out',output,'--workers',a.workers)
        run('audit_size_confirmation.py','--out',output);return
    if a.mode=='record-replay':
        run('final_record_replay.py','--out',root/'reproduced_final_replay','--workers',a.workers);return
    if a.mode=='figures':
        run('plot_ieee_all.py','--root',root);return
    if a.mode=='verify':
        subprocess.run([sys.executable,'-m','unittest','-v','test_theory','test_revision','test_round2','test_round3','test_round4','test_round5','test_mobility_cost','test_platform_cost','test_alignment'],cwd=code,env=env,check=True)
        run('audit_results.py');run('audit_revision.py','--root',root);run('audit_round2.py','--root',root);run('audit_round3.py','--root',root);run('audit_round4.py','--root',root);run('audit_round5.py','--root',root);run('audit_mobility_cost.py','--results',root/'mobility_cost_results');run('audit_platform_cost.py','--results',root/'platform_cost_results');run('audit_alignment.py','--results',root/'dense_alignment_results');return
    if a.mode=='equations':
        run('equation_code_audit.py','--output',root/'reproduced_equation_audit');return
    if a.mode=='alignment':
        out=root/'reproduced_alignment'/'dense_alignment_results'
        run('dense_alignment_study.py','--output',out,'--workers',a.workers)
        run('analyze_alignment.py','--results',out)
        run('alignment_sensitivity.py','--output',out,'--workers',a.workers)
        run('audit_alignment.py','--results',out,'--workers',a.workers)
        run('alignment_timing.py','--output',out);return
    if a.mode=='work':
        run('certificate_work_audit.py','--output',root/'reproduced_work'/'certificate_work_results');return
    if a.mode=='independent':
        run('independent_value_audit.py','--output',root/'reproduced_independent'/'independent_numerical_results');return
    if a.mode=='platform':
        out=root/'reproduced_platform'/'platform_cost_results'
        run('platform_cost_study.py','--output',out,'--workers',a.workers)
        run('analyze_platform_cost.py','--results',out)
        run('audit_platform_cost.py','--results',out);return
    if a.mode=='numerical':
        run('numerical_audit.py','--output',root/'reproduced_numerical'/'numerical_results');return
    if a.mode=='mobility':
        out=root/'reproduced_mobility'/'mobility_cost_results'
        run('mobility_cost_study.py','--output',out)
        run('audit_mobility_cost.py','--results',out);return
    if a.mode=='round5':
        target=root/'reproduced_round5';out=target/'round5_results'
        run('round5_experiments.py','--output',out)
        run('audit_round5.py','--root',root,'--results',out)
        run('plot_round5.py','--root',root,'--results',out,'--artifact-root',target)
        print('New R5 results:',target);return
    if a.mode=='round4':
        target=root/'reproduced_round4';out=target/'round4_results'
        run('physical_pilots.py','--output',out)
        run('audit_round4.py','--root',root,'--results',out)
        shutil.copytree(root/'manuscript',target/'manuscript',dirs_exist_ok=True)
        shutil.copytree(root/'figures',target/'figures',dirs_exist_ok=True)
        run('analyze_round4.py','--root',root,'--results',out,'--artifact-root',target)
        print('New pilot results:',out);return
    if a.mode=='round3':
        target=root/'reproduced_round3';out=target/'round3_results';out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'round3_results/protocol.json',out/'protocol.json')
        for mode in ['branch','vector','temporal','endpoint','catalogue']:
            run('round3_experiments.py','--mode',mode,'--root',root,'--output',out)
        run('finite_blocklength.py','--root',root,'--output',out)
        run('audit_round3.py','--root',root,'--results',out)
        shutil.copytree(root/'manuscript',target/'manuscript',dirs_exist_ok=True)
        shutil.copytree(root/'figures',target/'figures',dirs_exist_ok=True)
        run('analyze_round3.py','--root',root,'--results',out,'--artifact-root',target)
        print('New third-review results:',out);return
    if a.mode=='round2':
        target=root/'reproduced_round2';out=target/'round2_results';out.mkdir(parents=True,exist_ok=True)
        for mode in ['intervals','vector','calibration','fresh','timing','statistics']:
            run('round2_experiments.py','--mode',mode,'--root',root,'--output',out)
        run('audit_round2.py','--root',root,'--results',out)
        shutil.copytree(root/'manuscript',target/'manuscript',dirs_exist_ok=True)
        shutil.copytree(root/'figures',target/'figures',dirs_exist_ok=True)
        run('analyze_round2.py','--root',root,'--results',out,'--artifact-root',target)
        print('New second-review results:',out);return
    target=root/('smoke_run' if a.mode=='smoke' else 'reproduced');out=target/'revision_results';out.mkdir(exist_ok=True,parents=True)
    run('run_revision.py','--suite',a.mode,'--output',out,'--workers',a.workers)
    if a.mode=='smoke':print('Smoke results:',out);return
    for script in ['chance_benchmark.py','power_frontier.py','hybrid_frontier.py','screening_study.py','value_comparison.py']:
        run(script,'--output',out)
    run('ray_evaluation.py','--data',root/'ray_data','--output',out)
    run('conformal_report_filter.py','--data',root/'ray_data','--output',out)
    # No parallel jobs remain when dedicated sequential timing begins.
    run('latency_benchmark.py','--output',out)
    shutil.copytree(root/'manuscript',target/'manuscript',dirs_exist_ok=True)
    shutil.copytree(root/'figures',target/'figures',dirs_exist_ok=True)
    run('analyze_revision.py','--results',out,'--root',target)
    print('New results:',out)
    print('Review regenerated numbers before updating manuscript prose; measured timings vary.')

if __name__=='__main__':main()
