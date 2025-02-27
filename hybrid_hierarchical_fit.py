import pandas as pd
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt

model = CmdStanModel(stan_file="hybrid_hierarchical.stan")
df = pd.read_csv("weightnoisesim.csv")
df = df[df.noise == 'low']
assert len(df) == 200*200
model_dat = {
    'T': df.trial.max(), 'num_trials': [120, 200], 'N': len(df.participant.unique()),
    'choice1': [], 'choice2': [], 'state2': [], 'reward': [], 'condition': []
}
for i, (partnum, partdf) in enumerate(df.groupby('participant')):
    model_dat['choice1'].append(list(partdf.choice1))
    model_dat['choice2'].append(list(partdf.choice2))
    model_dat['state2'].append(list(partdf.state2))
    model_dat['reward'].append(list(partdf.reward))
    model_dat['condition'].append(int(i < 100) + 1)

fit = model.sample(
    data=model_dat, iter_warmup=8_000, iter_sampling=2_000, chains=4, adapt_delta=0.95, max_treedepth=12,
    refresh=1, show_progress=True, output_dir="hybrid_hierarchical_output")
print(fit.diagnose())

samples = fit.draws_pd()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Prior distribution')
plt.ylabel("Density")
plt.xlabel("Model-based weight")
plt.xlim(0, 1)
plt.hist(np.random.random(size=100_000), bins=100, density=True)
plt.subplot(1, 2, 2)
plt.title('Posterior distribution')
plt.ylabel("Density")
plt.xlabel("Model-based weight")
plt.xlim(0, 1)
plt.hist(samples["mu[1]"], bins=100, density=True, label="120 trials", alpha=0.5)
plt.hist(samples["mu[2]"], bins=100, density=True, label="200 trials", alpha=0.5)
plt.legend(loc="best", title="Condition")
plt.tight_layout()
plt.savefig("hybrid_hierarchical_results.pdf")
plt.close()