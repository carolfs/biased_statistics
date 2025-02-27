import pickle
import os
import pandas as pd
from cmdstanpy import CmdStanModel

weight_flnm = "three_stage_task_weights.bin"
data = pd.read_csv("exp1_converted.csv")
last_part = 0
if os.path.exists(weight_flnm):
    with open(weight_flnm, "rb") as inpf:
        for part, partdf in data.groupby("subject"):
            try:
                pickle.load(inpf)
            except:
                break
            else:
                last_part = part
if last_part < data.subject.max():
    novel_model = CmdStanModel(stan_file="three_stage_task_indiv.stan")
    with open(weight_flnm, "ab") as outf:
        for part, partdf in data.groupby("subject"):
            if part <= last_part:
                continue
            model_dat = {"num_trials": len(partdf), "higheff": partdf.high_effort, "station0": partdf.stim_0_1, "station1": partdf.stim_0_2, "choice0": partdf.choice0, "choice1": partdf.choice1, "state2": partdf.state2, "reward": partdf.points/9.}
            attempt = 1
            while True:
                print(f"Participant {part}, attempt {attempt}")
                fit = novel_model.sample(data=model_dat, iter_warmup=20_000, iter_sampling=50_000, chains=4, refresh=1, show_progress=True, adapt_delta=0.9)
                diag = fit.diagnose()
                print(diag)
                if sum(fit.divergences) == 0:
                    samples = fit.draws_pd()
                    pickle.dump({"wlow": samples["wlow"], "whtop": samples["whtop"], "whmid": samples["whmid"]}, outf)
                    break
                attempt += 1