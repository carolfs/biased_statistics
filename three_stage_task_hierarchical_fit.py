import argparse
from cmdstanpy import CmdStanModel
import pandas as pd

def fix_length(x):
    return list(x) + [-1]*(200 - len(x))

model = CmdStanModel(stan_file=f"three_stage_task_hierarchical_model.stan")
exp1 = pd.read_csv("exp1_converted.csv")
model_dat = {
    'num_subjects': len(exp1.subject.unique()),
    'num_trials': [len(trials) for _, trials in exp1.groupby('subject')],
    'higheff': [fix_length(trials.high_effort) for _, trials in exp1.groupby('subject')],
    'station0': [fix_length(trials.stim_0_1) for _, trials in exp1.groupby('subject')],
    'station1': [fix_length(trials.stim_0_2) for _, trials in exp1.groupby('subject')],
    'choice0': [fix_length(trials.choice0) for _, trials in exp1.groupby('subject')],
    'choice1': [fix_length(trials.choice1) for _, trials in exp1.groupby('subject')],
    'state2': [fix_length(trials.state2) for _, trials in exp1.groupby('subject')],
    'reward': [fix_length(trials.points/9.0) for _, trials in exp1.groupby('subject')],
}
parser = argparse.ArgumentParser()
parser.add_argument("iter_warmup", type=int)
parser.add_argument("iter_sampling", type=int)
parser.add_argument("chains", type=int)
args = parser.parse_args()
fit = model.sample(
    data=model_dat, show_progress=True, iter_warmup=args.iter_warmup, iter_sampling=args.iter_sampling, thin=1, refresh=1, chains=args.chains, output_dir="three_stage_task_hybrid_fit_output", adapt_delta=0.95, max_treedepth=12)
fit.summary().to_csv("hybrid_fit_summary.csv")
print(fit.diagnose())
