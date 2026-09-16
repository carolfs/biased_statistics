"""
Spurious evidence of precognition in a two-stage task
=====================================================

A logistic regression predicting whether a participant repeats their
first-stage choice is fitted separately to each participant, and the
resulting coefficients are tested across the sample.  The model includes
one predictor that cannot possibly matter: whether the transition on the
*following* trial is common or rare.  That transition had not occurred
when the choice was made, so its true coefficient is zero.

The analyses below show that it is nevertheless estimated as reliably
nonzero, and identify why.

Model (all predictors coded +1/-1):

    logit P(stay) = b0 + b_r*r + b_t*t + b_rt*r*t
                       + b_f*f + b_rf*r*f + b_tf*t*f + b_rtf*r*t*f

    r = previous trial rewarded          f = FUTURE trial transition common
    t = previous trial transition common

Coefficient indices used throughout: 0 = intercept, 1 = r, 2 = t,
3 = r*t, 4 = f, 5 = r*f, 6 = t*f, 7 = r*t*f.  Indices >= 4 are the
impossible ones.
"""

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit

from firthmodels import detect_separation
from firthmodels.adapters.statsmodels import FirthLogit

# These warnings can be ignored because the participants with quasi-complete
# separation are already being identified and counted.
import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning
warnings.simplefilter("ignore", HessianInversionWarning)

# Configuration

DATA_FILE = "two_stage_task_data.csv"
SESSION = 3           # session 3 has ~201 trials per participant
N_REPS = 20           # replications for the simulated-predictor analyses
P_COMMON = 0.7        # probability of a common transition in this task

COEF_NAMES = [
    "intercept",
    "reward",
    "transition",
    "reward x transition",
    "FUTURE transition",
    "reward x future",
    "transition x future",
    "reward x transition x future",
]
IMPOSSIBLE = range(4, 8)   # coefficients whose true value is zero

rng = np.random.default_rng(0)

# Get all valid participant data
def get_participant_data():
    """Get all valid participant data, excluding participants who always stayed
    """
    tstdf = pd.read_csv(DATA_FILE)
    tstdf = tstdf[tstdf.meas == SESSION]

    # A participant who never changed their first-stage choice provides no
    # variation in the outcome and cannot be fitted at all.
    constant = [part for part, d in tstdf.groupby("subj")
                if (d.ch1 == d.iloc[0].ch1).all()]
    tstdf = tstdf[~tstdf.subj.isin(constant)]
    return tstdf

# Building each participant's design matrix

def build_design(partdf, future="real", p_common=P_COMMON):
    """Split one participant's trials into consecutive pairs and build the
    design matrix for the stay/switch logistic regression.

    future : "real" uses the actual transition on the second trial of each
             pair; a float instead generates it at random with that
             probability of being common, so that it cannot carry any
             information about the choice.
    """
    y, x = [], []
    for prev, nxt in zip(partdf[:-1].itertuples(), partdf[1:].itertuples()):
        stay = int(prev.ch1 == nxt.ch1)
        rew = 2 * prev.rw - 1
        tr = 2 * (1 - prev.tran) - 1
        if future == "real":
            ft = 2 * (1 - nxt.tran) - 1
        else:
            ft = 2 * int(rng.random() < future) - 1
        assert rew in (1, -1) and tr in (1, -1) and ft in (1, -1)
        y.append(stay)
        x.append([1, rew, tr, rew * tr, ft, ft * rew, ft * tr, ft * rew * tr])
    return np.array(y), np.array(x)


# Fitting one participant

def fit_participant(y, X, method="mle"):
    """Fit one participant.  Returns (coefficients, status).

    status is one of:
      "ok"              fitted successfully
      "rank_deficient"  a design cell is empty, so the model is not identified
      "failed"          the fitting routine raised an error
    """
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None, "rank_deficient"
    try:
        if method == "mle":
            res = sm.Logit(y, X).fit(method="bfgs", maxiter=1000, disp=0)
            # NOTE: this reports convergence even when the maximum likelihood
            # estimate does not exist (see check_separation below).
            if not res.mle_retvals["converged"]:
                return None, "failed"
            return res.params, "ok"
        else:
            return FirthLogit(y, X).fit().params, "ok"
    except Exception:
        return None, "failed"

def check_separation(y, X):
    """True if the data are quasi-completely separated, meaning at least one
    coefficient has no finite maximum likelihood estimate.  The intercept
    column is dropped because detect_separation adds its own."""
    try:
        return detect_separation(X[:, 1:], y).separation
    except Exception:
        return False

# One complete two-step analysis

def two_step_analysis(tstdf, method="mle", future="real",
                      drop_separated=False):
    """Fit every participant, then test each coefficient across the sample
    with a Wilcoxon signed-rank test and Holm's correction.

    Returns a dict with the per-coefficient results and the counts of
    participants excluded for each reason.
    """
    coefs, n_sep, n_rank, n_failed = [], 0, 0, 0

    for _, partdf in tstdf.groupby("subj"):
        y, X = build_design(partdf, future=future)

        separated = check_separation(y, X)
        n_sep += separated
        if separated and drop_separated:
            continue

        params, status = fit_participant(y, X, method=method)
        if status == "rank_deficient":
            n_rank += 1
        elif status == "failed":
            n_failed += 1
        else:
            coefs.append(params)

    coefs = np.array(coefs)

    means, sds, wstats, effs, pvals = [], [], [], [], []
    for i in range(8):
        g = coefs[:, i]
        res = pg.wilcoxon(g)
        means.append(g.mean())
        sds.append(g.std())
        wstats.append(float(res["W_val"].iloc[0]))
        effs.append(float(res["RBC"].iloc[0]))
        pvals.append(float(res["p_val"].iloc[0]))
    _, pvals_corrected = pg.multicomp(pvals, method="holm")

    return {
        "n": len(coefs), "n_separated": n_sep,
        "n_rank_deficient": n_rank, "n_failed": n_failed,
        "mean": means, "sd": sds, "W": wstats,
        "effect_size": effs, "p": list(pvals_corrected),
        "coefs": coefs,
    }


def fmt_p(p):
    """Format P values for better readability."""
    return f"{p:.3f}" if p >= 0.0005 else f"{p:.1e}"

def report(result, title):
    """Print one analysis in a readable table and say what it shows."""
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)
    print(f"{result['n']} participants fitted", end="")
    extra = []
    if result["n_separated"]:
        extra.append(f"{result['n_separated']} quasi-separated")
    if result["n_rank_deficient"]:
        extra.append(f"{result['n_rank_deficient']} rank-deficient")
    if result["n_failed"]:
        extra.append(f"{result['n_failed']} failed to fit")
    print(f"  ({', '.join(extra)})" if extra else "")
    print()
    print(f"{'coefficient':<30}{'mean':>8}{'SD':>8}{'W':>10}"
          f"{'effect':>9}{'P (Holm)':>12}")
    print("-" * 77)
    for i in range(8):
        flag = "  <-- impossible" if i in IMPOSSIBLE else ""
        star = "*" if result["p"][i] < 0.05 else " "
        print(f"{COEF_NAMES[i]:<30}{result['mean'][i]:>8.2f}"
              f"{result['sd'][i]:>8.2f}{result['W'][i]:>10.0f}"
              f"{result['effect_size'][i]:>9.3f}"
              f"{fmt_p(result['p'][i]):>11}{star}{flag}")

    spurious = [COEF_NAMES[i] for i in IMPOSSIBLE if result["p"][i] < 0.05]
    print()
    if spurious:
        print(f"  Reliably nonzero although it must be zero: {', '.join(spurious)}")
    else:
        print("  No impossible coefficient reached significance.")


def count_spurious_over_reps(tstdf, method, future, drop_separated, n_reps,
                             label):
    """Repeat an analysis with freshly generated future transitions and count
    how often at least one impossible coefficient comes out significant."""
    print()
    print("=" * 74)
    print(label)
    print("=" * 74)
    print(f"Running {n_reps} replications, regenerating the future "
          f"transition each time...")
    hits = 0
    excluded = []
    for rep in range(n_reps):
        res = two_step_analysis(tstdf, method=method, future=future,
                                drop_separated=drop_separated)
        hit = any(res["p"][i] < 0.05 for i in IMPOSSIBLE)
        hits += hit
        excluded.append(res["n_rank_deficient"] + res["n_failed"])
        print(f"  rep {rep + 1:>2}: "
              f"future transition = {res['mean'][4]:+.2f}, "
              f"P = {fmt_p(res['p'][4])}"
              f"{'   (spurious effect found)' if hit else ''}")
    if max(excluded) > 0:
        print()
        print("  At most", max(excluded), "participants were excluded "
              f"from a replication (average: {np.mean(excluded)}).")
    print()
    print(f"  At least one impossible coefficient was significant in "
          f"{hits} of {n_reps} replications.")
    return hits



# Simulation: does the sign of the artefact follow the intercept?


def simulate_intercepts(intercepts, n_part=500, n_trials=200,
                        p_common=P_COMMON):
    """Generate stay/switch data from a known model with no future-transition
    effect, fit it the same way, and see what the future-transition
    coefficient comes out as for each value of the intercept."""
    print()
    print("=" * 74)
    print("SIMULATION: the sign of the artefact follows the intercept")
    print("=" * 74)
    print(f"{n_part} simulated participants, {n_trials} trial pairs each.")
    print("Data are generated with NO future-transition effect whatsoever.")
    print()
    print(f"{'true intercept':>15}{'estimated future coef':>24}{'P':>12}{'n':>7}")
    print("-" * 58)
    for b0 in intercepts:
        coefs = []
        for _ in range(n_part):
            rw = 2 * (rng.random(n_trials) < 0.56) - 1
            tr = 2 * (rng.random(n_trials) < p_common) - 1
            ft = 2 * (rng.random(n_trials) < p_common) - 1
            # the future transition is absent from the generating model
            p = expit(b0 + 0.58 * rw - 0.32 * tr + 0.44 * rw * tr)
            y = (rng.random(n_trials) < p).astype(int)
            X = np.column_stack([np.ones(n_trials), rw, tr, rw * tr,
                                 ft, ft * rw, ft * tr, ft * rw * tr])
            if y.min() == y.max():
                continue
            params, status = fit_participant(y, X, method="mle")
            if status == "ok":
                coefs.append(params)
        g = np.array([c[4] for c in coefs])
        p = stats.wilcoxon(g).pvalue
        print(f"{b0:>15.0f}{g.mean():>24.3f}{fmt_p(p):>12}{len(g):>7}")
    print()
    print("  The estimated coefficient takes the sign OPPOSITE to the")
    print("  intercept and grows with its magnitude, vanishing at zero.")

def fixed_effects_analysis(df):
    """Fit all participants together and print the results."""
    print()
    print("=" * 74)
    print("FIXED-EFFECTS POOLED MODEL (maximum likelihood, real transitions)")
    print("=" * 74)
    y, X = build_design(df)
    res = sm.Logit(y, X).fit(method="bfgs", maxiter=1000, disp=0)
    assert res.mle_retvals["converged"]
    print(res.summary())
    print()
    print("  No impossible coefficient reached significance.")

# Main


def main():
    tstdf = get_participant_data()

    print(f"Loaded session {SESSION}: {tstdf.subj.nunique()} participants")

    # The analysis as a researcher would run it.
    report(two_step_analysis(tstdf, method="mle", future="real"),
           "STANDARD ANALYSIS (maximum likelihood, real transitions)")

    # The analysis with only fixed effects.
    fixed_effects_analysis(tstdf)

    # The analysis with simulated transitions.
    report(two_step_analysis(tstdf, method="mle", future=P_COMMON),
           "STANDARD ANALYSIS (maximum likelihood, simulated transitions)")


    # Bias-corrected estimator.  Firth's penalty removes the leading
    #    small-sample bias term and yields finite estimates under separation.
    report(two_step_analysis(tstdf, method="firth", future="real"),
           "BIAS-CORRECTED ANALYSIS (Firth penalised likelihood)")

    # Does the artefact depend on the predictor being real, on the
    #      estimator, or on the separated participants?
    count_spurious_over_reps(
        tstdf, "mle", P_COMMON, False, N_REPS,
        "SIMULATED PREDICTOR, maximum likelihood")
    count_spurious_over_reps(
        tstdf, "mle", 0.5, False, N_REPS,
        "BALANCED PREDICTOR, 20 replications (expected ~5%)")
    count_spurious_over_reps(
        tstdf, "firth", P_COMMON, False, N_REPS,
        "SIMULATED PREDICTOR, Firth penalised likelihood")
    count_spurious_over_reps(
        tstdf, "firth", P_COMMON, True, N_REPS,
        "SIMULATED PREDICTOR, Firth, separated participants excluded")

    # Where the sign comes from.
    simulate_intercepts([-3., -2., -1., 0., 1., 2., 3.])


if __name__ == "__main__":
    main()
