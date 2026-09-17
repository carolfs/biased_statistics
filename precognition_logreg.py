"""Spurious evidence of precognition in a two-stage task: shared code.

A logistic regression predicting whether a participant repeats their first-stage
choice is fitted separately to each participant, and the resulting coefficients
are tested across the sample.  The model includes one predictor that cannot
possibly matter: whether the transition on the *following* trial is common or
rare.  That transition had not occurred when the choice was made, so its true
coefficient is zero.

This module holds the parts that more than one analysis needs.
See precognition.qmd for the exposition and hier_logreg_fit.py for
the hierarchical Bayesian fit.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm

from firthmodels import detect_separation
from firthmodels.adapters.statsmodels import FirthLogit
from statsmodels.tools.sm_exceptions import HessianInversionWarning

# Participants with quasi-complete separation are identified and counted
# explicitly, so the Hessian warnings they raise add nothing.
warnings.simplefilter("ignore", HessianInversionWarning)


# Configuration

DATA_FILE = Path("two_stage_task_data.csv")
SESSION = 3        # session 3 has the most trials per participant (201)
P_COMMON = 0.7     # probability of a common transition in this task

COEF_NAMES = [
    "intercept",
    "reward",
    "previous transition",
    "reward:previous transition",
    "future transition",
    "reward:future transition",
    "previous transition:future transition",
    "reward:previous transition:future transition",
]
IMPOSSIBLE = range(4, 8)   # coefficients whose true value is zero

SEED = 0
rng = np.random.default_rng(SEED)


def set_seed(seed=SEED):
    """Reseed this module's shared generator."""
    global rng
    rng = np.random.default_rng(seed)
    return rng


# Counting the fits

FITS = {"total": 0}   # every model fit actually attempted
FIT_LOG = []          # (label, number of fits) per analysis


def note_fits(label):
    """Attribute the fits performed since the last call to one analysis."""
    already_attributed = sum(n for _, n in FIT_LOG)
    FIT_LOG.append((label, FITS["total"] - already_attributed))


def reset_fits():
    """Forget the fits counted so far."""
    FITS["total"] = 0
    FIT_LOG.clear()


# Loading the data


def get_participant_data(path=DATA_FILE, session=SESSION):
    """Load one session and drop participants with no variation in the outcome.

    Returns the trial-level DataFrame.  Two counts are attached to df.attrs:
    "n_in_file", the number of participants the session contains, and
    "excluded_constant", the identifiers of those dropped here.
    """
    df = pd.read_csv(path)
    df = df[df.meas == session]
    n_in_file = df.subj.nunique()

    # A participant who never changed their first-stage choice provides no
    # variation in the outcome and cannot be fitted by any method used here.
    constant = [s for s, d in df.groupby("subj")
                if (d.ch1 == d.iloc[0].ch1).all()]
    df = df[~df.subj.isin(constant)]
    df.attrs["n_in_file"] = n_in_file
    df.attrs["excluded_constant"] = constant
    return df


# Building each participant's design matrix


def build_design(partdf, future="real"):
    """Split one participant's trials into consecutive pairs and build the
    design matrix for the stay/switch logistic regression.

    future : "real" uses the actual transition on the second trial of each
             pair; a float instead generates that predictor at random, with
             that probability of being common, so that it cannot carry any
             information whatsoever about the choice.
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
    """Fit one participant, assuming X has already been checked for full column
    rank.  Returns (coefficients, status), where status is "ok" or "failed"."""
    FITS["total"] += 1
    try:
        if method == "mle":
            res = sm.Logit(y, X).fit(method="bfgs", maxiter=1000, disp=0)
            # NOTE: this reports convergence even when the maximum likelihood
            # estimate does not exist (see participant_status below).
            if not res.mle_retvals["converged"]:
                return None, "failed"
            return res.params, "ok"
        else:
            return FirthLogit(y, X).fit().params, "ok"
    except Exception:
        return None, "failed"


def participant_status(y, X, method="mle", drop_separated=False,
                       check_sep=True):
    """Decide whether one participant can be fitted, and fit them if so.

    This is the only place the design matrix is screened.  Returns
    (coefficients, status, separated), where status is "ok", "dropped",
    "rank_deficient" or "failed", and separated says whether the data are
    quasi-completely separated.  The intercept column is dropped before the
    separation test because detect_separation adds its own.
    """
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None, "rank_deficient", False   # model not identified, so
                                               # nothing is fitted and no
                                               # separation test is possible
    if check_sep or drop_separated:
        separated = bool(detect_separation(X[:, 1:], y).separation)
    else:
        separated = False
    if separated and drop_separated:
        return None, "dropped", True
    params, status = fit_participant(y, X, method=method)
    return params, status, separated


# One complete two-step analysis


def two_step_analysis(tstdf, method="mle", future="real",
                      drop_separated=False, check_sep=True):
    """Fit every participant, then test each coefficient across the sample
    with a Wilcoxon signed-rank test and Holm's correction.

    Returns a dict with the per-coefficient results and the counts of
    participants excluded for each reason.
    """
    coefs, n_sep, n_rank, n_failed = [], 0, 0, 0

    for _, partdf in tstdf.groupby("subj"):
        y, X = build_design(partdf, future=future)

        params, status, separated = participant_status(
            y, X, method=method, drop_separated=drop_separated,
            check_sep=check_sep)
        n_sep += separated
        if status == "rank_deficient":
            n_rank += 1
        elif status == "failed":
            n_failed += 1
        elif status == "ok":
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
    return f"{p:.3f}" if p >= 0.0005 else "<0.001"
