"""Compare the two-step and hierarchical estimates of the same coefficients.

Draws precognition_effect_sizes.png, the figure in which each of the eight
coefficients of the logistic regression model appears twice: once as the mean
of the participant-level maximum likelihood estimates, with a bootstrap
confidence interval, and once as the population-level coefficient of the
hierarchical Bayesian model, with a credible interval.  Every coefficient
involving the future transition has a true value of zero; the two-step
estimates of those coefficients are displaced from zero and the hierarchical
ones are not, and every hierarchical estimate is smaller in magnitude than its
two-step counterpart.

Run `python hier_logreg_fit.py` first: this script reads the draws it left in
precognition_hierarchical_results and does not sample.
"""
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import from_csv
from scipy import stats
from precognition_logreg import get_participant_data, two_step_analysis

def main():
    # Run the two-step analysis
    df = get_participant_data()
    # coefs is one row per participant, one column per coefficient.
    coefs = two_step_analysis(df, method="mle", future="real")["coefs"]

    # Plot confidence intervals along with Bayesian analysis results
    bayesian_results_flnm = "precognition_hierarchical_results"
    samples = from_csv(bayesian_results_flnm).draws_pd()
    plt.figure(figsize=(8, 5))
    means = []
    lows = []
    highs = []
    # Two-step: the mean across participants, with a 95% bootstrap interval
    # (BCa by default) around it.
    for i in range(8):
        s = [c[i] for c in coefs]
        b = stats.bootstrap([s], np.mean)
        means.append(np.mean(s))
        lows.append(b.confidence_interval.low)
        highs.append(b.confidence_interval.high)
    # errorbar wants distances from the point, not the interval limits.
    # Offset by -0.1 so the two methods do not overlap at each tick.
    plt.errorbar(np.arange(8) - 0.1, means, yerr=[
        [m - l for m, l in zip(means, lows)],
        [h - m for m, h in zip(means, highs)]], fmt='o',
        label="Multi-stage analysis")
    means = []
    lows = []
    highs = []
    # Hierarchical: the population-level coefficients mu, with the central 95%
    # of the posterior draws.  Stan arrays are 1-indexed, hence mu[i + 1].
    for i in range(8):
        mu = samples[f"mu[{i + 1}]"]
        means.append(mu.mean())
        lows.append(mu.quantile(0.025))
        highs.append(mu.quantile(0.975))
    plt.errorbar(np.arange(8) + 0.1, means, yerr=[
        [m - l for m, l in zip(means, lows)],
        [h - m for m, h in zip(means, highs)]], fmt='o',
    label="Bayesian hierarchical model")
    # Tick labels in the same order as the columns of the design matrix.
    plt.xticks(np.arange(8), ["intercept", "$r$", "$t$", r"$r \times t$", "$f$",
                              r"$r \times f$", r"$t \times f$",
                              r"$r \times t \times f$"])
    plt.xlabel("Coefficients")
    plt.ylabel("Estimates (mean, 95% CI)")
    plt.legend(loc="best", title="Method")
    # The intercept is far above the rest, so the lower limit is fixed and the
    # upper one left to matplotlib.
    plt.ylim(-1.0, None)
    plt.grid(axis="y")
    plt.savefig("precognition_effect_sizes.png")
    plt.close()

if __name__ == "__main__":
    main()
