import os
import shutil
import random
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel, from_csv
from pandas.core.groupby import groupby

def get_converged_samples(fit):
    if sum(fit.divergences) > 0:
        # Filter out chains with divergences
        fit.save_csvfiles("chains")
        files = []
        for div, chain in zip(fit.divergences, fit.chain_ids):
            if div > 0:
                continue
            for flnm in os.listdir("chains"):
                if flnm.endswith(f"_{chain}.csv"):
                    break
            else:
                raise Exception("Could not find chain file")
            files.append(os.path.join("chains", flnm))
        if len(files) < 3:
            rt = None
        else:
            fit = from_csv(files)
            assert sum(fit.divergences) == 0
            rt = fit.draws_pd()
        shutil.rmtree("chains")
        return rt
    else:
        return fit.draws_pd()

model = CmdStanModel(stan_file="hybrid_single.stan")
tstdata = pd.read_csv("two_stage_task_data.csv")

splitdata = [measdata for _, measdata in tstdata.groupby(["subj", "meas"])]
random.seed(284853)
random.shuffle(splitdata)

NUMWARMUP = 10_000
NUMSAMPLES = 10_000
plt.figure(figsize=(2*5, 5*3))
num_subjs = 1
samples_flnm = "tst-subj{}-s{}-samples.csv"
analysed = [0 for _ in splitdata]
for i, subjdf in enumerate(splitdata):
    flnm = samples_flnm.format(subjdf.subj.iloc[0], subjdf.meas.iloc[0])
    if os.path.exists(flnm):
        analysed[i] = 1

for subjdf in splitdata:
    if subjdf.meas.iloc[0] != 1:
        continue
    flnm = samples_flnm.format(subjdf.subj.iloc[0], subjdf.meas.iloc[0])
    if os.path.exists(flnm):
        samples = pd.read_csv(flnm)
        print("Recovered", num_subjs, "from", flnm)
    elif sum(analysed) > 10:
        continue
    else:
        model_dat = {'T': len(subjdf), 'choice1': subjdf.ch1, 'choice2': subjdf.ch2, 'state2': subjdf.state, 'reward': subjdf.rw}
        fit = model.sample(data=model_dat, iter_warmup=NUMWARMUP, iter_sampling=NUMSAMPLES, chains=6, adapt_delta=0.95, show_progress=True)
        samples = get_converged_samples(fit)
        if samples is None:
            print("Skip participant")
            continue
        samples.to_csv(flnm, index=False)
    plt.subplot(5, 2, num_subjs)
    plt.title(f"Participant {subjdf.subj.iloc[0]}, session {subjdf.meas.iloc[0]} ({len(subjdf)} trials)")
    plt.hist(samples["w"], bins=50, density=True)
    plt.xlim(0, 1)
    plt.xlabel("Model-based weight")
    plt.ylabel("Probability density")
    num_subjs += 1
    if num_subjs > 5:
        break
for subjdf in splitdata:
    if len(subjdf) < 200:
        continue
    flnm = samples_flnm.format(subjdf.subj.iloc[0], subjdf.meas.iloc[0])
    if os.path.exists(flnm):
        samples = pd.read_csv(flnm)
        print("Recovered", num_subjs, "from", flnm)
    elif sum(analysed) > 10:
        continue
    else:
        model_dat = {'T': len(subjdf), 'choice1': subjdf.ch1, 'choice2': subjdf.ch2, 'state2': subjdf.state, 'reward': subjdf.rw}
        fit = model.sample(data=model_dat, iter_warmup=NUMWARMUP, iter_sampling=NUMSAMPLES, chains=6, adapt_delta=0.95, show_progress=True)
        samples = get_converged_samples(fit)
        if samples is None:
            print("Skip participant")
            continue
        samples.to_csv(flnm, index=False)
    plt.subplot(5, 2, num_subjs)
    plt.title(f"Participant {subjdf.subj.iloc[0]}, session {subjdf.meas.iloc[0]} ({len(subjdf)} trials)")
    plt.hist(samples["w"], bins=50, density=True)
    plt.xlim(0, 1)
    plt.xlabel("Model-based weight")
    plt.ylabel("Probability density")
    num_subjs += 1
    if num_subjs > 10:
        break
plt.tight_layout()
plt.savefig("ex_mb_weight_distr.pdf")
plt.close()

samples_flnm = "tst-simpart{}-samples.csv"
simdata = pd.read_csv("weightnoisesim.csv")
plt.figure(figsize=(2*5, 4*3))
subplot = 1
for noisecon in ("high", "low"):
    condf = simdata[simdata.noise == noisecon]
    for numtrials, (_, subjdf) in zip((120, 120, 200, 200), condf.groupby("participant")):
            plt.subplot(4, 2, subplot)
            subjdf = subjdf[:numtrials]
            flnm = samples_flnm.format(subjdf.participant.iloc[0])
            if os.path.exists(flnm):
                samples = pd.read_csv(flnm)
            else:
                model_dat = {
                'T': len(subjdf), 'choice1': subjdf.choice1, 'choice2': subjdf.choice2,
                'state2': subjdf.state2, 'reward': subjdf.reward}
                fit = model.sample(data=model_dat, iter_warmup=NUMWARMUP, iter_sampling=NUMSAMPLES, chains=6, adapt_delta=0.95, show_progress=True)
                samples = get_converged_samples(fit)
                if samples is None:
                    print("Skip participant")
                    continue
                samples.to_csv(flnm, index=False)
            subplot += 1
            plt.title(f"{noisecon.capitalize()}-noise agent ({len(subjdf)} trials)")
            plt.hist(samples["w"], bins=50, density=True)
            plt.xlim(0, 1)
            plt.xlabel("Model-based weight")
            plt.ylabel("Probability density")
plt.tight_layout()
plt.savefig("ex_sim_mb_weight_distr.pdf")
plt.close()
