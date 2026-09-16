import os
import pandas as pd
from cmdstanpy import CmdStanModel, from_csv
import matplotlib.pyplot as plt
from precognition_logreg import get_participant_data, build_design

OUTPUT_DIR = "precognition_hierarchical_results"
N = 200
K = 8

def main():
    if not os.path.exists(OUTPUT_DIR):
        model = CmdStanModel(stan_file="hier_logreg_model.stan")
        df = get_participant_data()
        model_dat = {
            'M': len(df.subj.unique()),
            'N': N,
            'K': 8,
            'y': [],
            'x': [],
        # array[M, N] int<lower=0, upper=1> y; // stay for each trial
        # array[M] matrix[N, K] x; // group predictors for each trial
        }
        for i, (partnum, partdf) in enumerate(df.groupby('subj')):
            y, x = build_design(partdf)
            assert len(x) == len(y)
            assert len(y) <= N
            y = list(y) + [0]*(N - len(y))
            x = list(x) + [[0.]*K]*(N - len(x))
            model_dat['y'].append(y)
            model_dat['x'].append(x)

        fit = model.sample(
            data=model_dat, iter_warmup=1_000, iter_sampling=2_000, chains=5, adapt_delta=0.9, max_treedepth=12,
            refresh=1, show_progress=True, output_dir=OUTPUT_DIR)
        print(fit.diagnose())
        plot_results(fit.draws_pd())
    else:
        samples = from_csv(OUTPUT_DIR).draws_pd()
        plot_results(samples)

if __name__ == "__main__":
    main()
