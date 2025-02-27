from os.path import exists
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import pingouin as pg
import matplotlib.pyplot as plt
from cmdstanpy import from_csv

tstdf = pd.read_csv("two_stage_task_data.csv")
tstdf = tstdf[tstdf.meas == 3]

coefs = []
failed = []
outf = open("stay_two_stage_task.csv", "w")
outf.write("participant,stay,reward,previous_transition,next_transition\n")
for part, partdf in tstdf.groupby("subj"):
    y, x = [], []
    for prevtrial, nexttrial in zip(partdf[:-1].itertuples(), partdf[1:].itertuples()):
        assert prevtrial.trl < nexttrial.trl
        y.append(int(prevtrial.ch1 == nexttrial.ch1))
        rew = 2*(prevtrial.rw) - 1
        prev_tr = 2*(1 - prevtrial.tran) - 1
        next_tr = 2*(1 - nexttrial.tran) - 1
        assert rew in (1, -1)
        assert prev_tr in (1, -1)
        assert next_tr in (1, -1)
        xrow = np.array([1, rew, prev_tr, rew*prev_tr, next_tr, rew*next_tr, prev_tr*next_tr, rew*prev_tr*next_tr])
        x.append(xrow)
        outf.write(f"part{part},{y[-1]},{rew},{prev_tr},{next_tr}\n")
    try:
        model = sm.Logit(y, x)
        results = model.fit(method="bfgs", maxiter=1_000, disp=0)
    except:
        failed.append(part)
        continue
    else:
        assert results.mle_retvals["converged"]
        coefs.append(results.params)
outf.close()
pvals = []
for i in range(8):
    g = np.array([p[i] for p in coefs])
    testresults = pg.wilcoxon(g)
    pvals.append(float(testresults["p-val"].iloc[0]))
    print(round(np.mean(g), 2), round(np.std(g), 2), round(testresults['W-val'].iloc[0], 3), round(float(testresults["RBC"].iloc[0]), 3),
        sep="\t")
print("Multiple comparison results:")
for significant, p_value in zip(*pg.multicomp(pvals)):
    print(significant, p_value)
if len(failed):
    print("Failed convergence for participants", *failed)

# Plot confidence intervals along with Bayesian analysis results
bayesian_results_flnm = "precognition_hierarchical_results"
try:
    samples = from_csv(bayesian_results_flnm).draws_pd()
    plt.figure(figsize=(8, 5))
    means = []
    lows = []
    highs = []
    for i in range(8):
        s = [c[i] for c in coefs]
        b = stats.bootstrap([s], np.mean)
        means.append(np.mean(s))
        lows.append(b.confidence_interval.low)
        highs.append(b.confidence_interval.high)
    plt.errorbar(np.arange(8) - 0.1, means, yerr=[[m - l for m, l in zip(means, lows)], [h - m for m, h in zip(means, highs)]], fmt='o', label="Multi-stage analysis")
    means = []
    lows = []
    highs = []
    for i in range(8):
        mu = samples[f"mu[{i + 1}]"]
        means.append(mu.mean())
        lows.append(mu.quantile(0.025))
        highs.append(mu.quantile(0.975))
    plt.errorbar(np.arange(8) + 0.1, means, yerr=[[m - l for m, l in zip(means, lows)], [h - m for m, h in zip(means, highs)]], fmt='o', label="Bayesian hierarchical model")
    plt.xticks(np.arange(8), ["intercept", "$r$", "$t$", r"$r \times t$", "$f$", r"$r \times f$", r"$t \times f$", r"$r \times t \times f$"])
    plt.xlabel("Coefficients")
    plt.ylabel("Estimates (mean, 95% CI)")
    plt.legend(loc="best", title="Method")
    plt.ylim(-1.0, None)
    plt.grid(axis="y")
    plt.savefig("precognition_effect_sizes.png")
    plt.close()
except:
    print("Bayesian analysis failed!")

# Analysis using fake future transition data

nexttr_effect = 0
num_reps = 20
for rep in range(num_reps):
    for part, partdf in tstdf.groupby("subj"):
        y, x = [], []
        for prevtrial, nexttrial in zip(partdf[:-1].itertuples(), partdf[1:].itertuples()):
            assert prevtrial.trl < nexttrial.trl
            y.append(int(prevtrial.ch1 == nexttrial.ch1))
            rew = 2*(prevtrial.rw) - 1
            prev_tr = 2*(1 - prevtrial.tran) - 1
            next_tr = 2*(np.random.random() < 0.7) - 1
            assert rew in (1, -1)
            assert prev_tr in (1, -1)
            assert next_tr in (1, -1)
            xrow = np.array([1, rew, prev_tr, rew*prev_tr, next_tr, rew*next_tr, prev_tr*next_tr, rew*prev_tr*next_tr])
            x.append(xrow)
        try:
            model = sm.Logit(y, x)
            results = model.fit(method="bfgs", maxiter=1_000_000, disp=0)
        except:
            failed.append(part)
            continue
        else:
            assert results.mle_retvals["converged"]
            coefs.append(results.params)
    pvals = []
    for i in range(8):
        g = np.array([p[i] for p in coefs])
        testresults = pg.wilcoxon(g)
        pvals.append(float(testresults["p-val"].iloc[0]))
        if rep == 1:
            print(round(np.mean(g), 2), round(np.std(g), 2), round(testresults['W-val'].iloc[0], 3), round(float(testresults["RBC"].iloc[0]), 3),
                sep="\t")
    if rep == 1:
        print("Multiple comparison results:")
    for i, (significant, p_value) in enumerate(zip(*pg.multicomp(pvals))):
        if rep == 1:
            print(significant, p_value)
        if i == 5: # Next transition effect
            if p_value < 0.05:
                nexttr_effect += 1
    if len(failed):
        print("Failed convergence for participants", *failed)
print("Found future transition effect", nexttr_effect, "times ouf of", num_reps)
