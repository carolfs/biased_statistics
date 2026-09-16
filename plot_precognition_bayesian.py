import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import from_csv
from scipy import stats
from precognition_logreg import get_participant_data, two_step_analysis

# Run the two-step analysis
df = get_participant_data()
coefs = two_step_analysis(df, method="mle", future="real")["coefs"]

# Plot confidence intervals along with Bayesian analysis results
bayesian_results_flnm = "precognition_hierarchical_results"
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

