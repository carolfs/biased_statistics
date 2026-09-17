# Spurious evidence of precognition in a two-stage task
Carolina Feher da Silva
2026-09-17

## Introduction

A logistic regression predicting whether a participant repeats their
choice in consecutive trials is fitted separately to each participant,
and the resulting coefficients are tested across the sample. The model
includes one predictor that cannot possibly have an effect on behaviour:
whether the state transition on the *following* trial is common or rare.
That transition had not occurred when the choice was made, and it is
independent of everything that preceded it, so its true coefficient is
zero.

It is nevertheless estimated as consistently nonzero across 563
participants. This document reproduces that result, establishes that it
is not caused by the predictor being derived from real data, by the
participants whose data are degenerate, or by the choice of estimator,
and identifies the mechanism: the predictor divides each participant’s
trials into two subsets of unequal size, the small-sample bias of
maximum likelihood is larger in the smaller subset, and the coefficient
records the difference between the two biases.

Every numerical claim made about this example in the manuscript is
produced by one of the cells below, with the exception of the
hierarchical Bayesian fit, which is produced by running
`python hier_logreg_fit.py`.

## Before you start

### Getting the data

The dataset is from Shahar et al. (2019) and is not distributed in this
repository. Two steps are required before the code below will run.

1.  **Download `mytst.mat`** from the OSF repository that accompanies
    Shahar et al. (2019), <https://osf.io/7dekj/>, and place it in the
    same directory as this document. The file is a MATLAB structure
    array containing one entry per participant, with the trial-by-trial
    data for all three sessions.

2.  **Convert it to CSV** by running the preprocessing script supplied
    alongside this document:

    ``` bash
    python preprocess_two_stage_task_data.py
    ```

    The script reads `mytst.mat`, flattens the MATLAB structure into a
    single table with one row per trial, **discards aborted trials**
    (those on which the participant failed to respond within the task’s
    time limit, marked by `abort == 1`), and writes
    `two_stage_task_data.csv` into the current directory. That CSV is
    the only input this document needs.

Because aborted trials are removed, two rows that are adjacent in the
CSV are not always two consecutive trials in the experiment, and the
number of trial pairs available varies slightly from participant to
participant. This, together with the exclusion of the participants
described below, is why the total number of pairs analysed is smaller
than $568 \times 200$.

> [!NOTE]
>
> ### Columns used
>
> `subj` (participant identifier), `meas` (session, 1–3), `ch1`
> (first-stage choice), `rw` (rewarded: 1, unrewarded: 0), `tran`
> (transition type; `0` is common and `1` is rare, which is why the
> recoding below inverts it), and `abort`, which the preprocessing
> script has already used.

### Software

Python 3.14 with `numpy`, `pandas`, `scipy`, `statsmodels` 0.15.0,
`pingouin`, and `firthmodels` 0.8.2. The file `precognition_logreg.py`,
which holds the analysis code this document imports, must sit in the
same directory. Additionally, regenerating this document with all of the
results requires Quarto and a Jupyter kernel:

``` bash
quarto render precognition.qmd
```

### Run time

The full document performs about 45,571 logistic regression fits and
takes less than 10 minutes to run on a modest laptop; the cells
performing repeated analyses are the slowest. Change `N_REPS` to 2–3 to
get faster results. The fits are counted as they are performed and
broken down by analysis in
<a href="#sec-fit-count" class="quarto-xref">Section 12</a>.

## Setup

``` python
import inspect

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit

import precognition_logreg as pl
from precognition_logreg import (
    COEF_NAMES, DATA_FILE, FITS, FIT_LOG, IMPOSSIBLE, P_COMMON, SESSION,
    build_design, fit_participant, fmt_p, get_participant_data, note_fits,
    participant_status, two_step_analysis,
)

N_REPS = 20        # replications for the simulated-predictor analyses

# This is just to allow caching while developing this document.
# Results are extremely similar with any seed.

SEED = 0
rng = pl.set_seed(SEED)


def show_source(*funcs):
    """Print the source of functions defined in precognition_logreg.py, so
    that the code under discussion appears here as well as in the module."""
    print("\n\n".join(inspect.getsource(f).rstrip() for f in funcs))


if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"{DATA_FILE} not found. Download mytst.mat from https://osf.io/7dekj/ "
        "and run `python preprocess_two_stage_task_data.py` first."
    )
```

## The task and the model

In each trial of the two-stage task (Daw et al., 2011), participants
choose between two first-stage options. Each option leads to one of two
second-stage states: to its own likely destination with probability 0.7
(a *common* transition) and to the other with probability 0.3 (a *rare*
transition). The participant then chooses again and is rewarded or not.

The standard analysis splits each participant’s trial sequence into
pairs of consecutive trials, codes the first-stage decision in each pair
as a *stay* if the participant repeated their previous first-stage
choice, and regresses it on the first trial’s outcome $x_r$ ($+1$
rewarded, $-1$ unrewarded) and transition type $x_t$ ($+1$ common, $-1$
rare):

<span id="eq-usual">$$\mathrm{logit}(P(\mathrm{stay})) = \beta_0 + \beta_r x_r + \beta_t x_t + \beta_{r\times t} x_r x_t. \qquad(1)$$</span>

To this model, I add the *second* trial’s transition, $x_f$, where $f$
stands for “future”, together with all of its interactions:

<span id="eq-full">$$\mathrm{logit}(P(\mathrm{stay})) = \beta_0 + \beta_r x_r + \beta_t x_t + \beta_{r\times t} x_r x_t + \beta_f x_f + \beta_{r\times f} x_r x_f + \beta_{t\times f} x_t x_f + \beta_{r\times t\times f} x_r x_t x_f. \qquad(2)$$</span>

The stay-or-switch decision in each pair is made *before* the second
trial’s transition occurs, and each transition is an independent random
event with fixed probabilities. Every coefficient involving $x_f$
therefore has a true value of zero.

### Loading the data

``` python
show_source(get_participant_data)

tst = get_participant_data()

trials_per_part = tst.groupby("subj").size()
print(f"Session {SESSION}: {tst.attrs['n_in_file']} participants in the file")
print("Excluded for making the same first-stage choice throughout: "
      f"{len(tst.attrs['excluded_constant'])}")
print(f"Retained: {tst.subj.nunique()} participants")
print(f"Trials per participant: median {trials_per_part.median():.0f}, "
      f"range {trials_per_part.min()}-{trials_per_part.max()}")
```

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
    Session 3: 568 participants in the file
    Excluded for making the same first-stage choice throughout: 5
    Retained: 563 participants
    Trials per participant: median 200, range 94-201

### Building each participant’s design matrix

``` python
show_source(build_design)
```

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

### Fitting one participant

The data loading, the design matrix, the fitting and the two-step
analysis live in `precognition_logreg.py`, so that `hier_logreg_fit.py`
can use the same code; each is listed here as it is introduced.

`participant_status` is the gatekeeper: it screens the design matrix,
runs the separation test when one is wanted, and delegates the fitting
itself to `fit_participant`, which assumes the matrix has already been
checked. A design matrix is rank deficient when one of the eight
combinations of $x_r$, $x_t$ and $x_f$ is absent from that participant’s
trial pairs, in which case the model is not identified, nothing is
fitted, and the separation test cannot be run either.

``` python
show_source(fit_participant, participant_status)
```

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

### One complete two-step analysis

Each participant is fitted separately, and each coefficient is then
tested against zero across participants with a two-sided Wilcoxon
signed-rank test, with $P$ values corrected for multiple comparisons by
Holm’s method. The effect size is the matched-pairs rank-biserial
correlation. This is a typical two-step analysis.

``` python
show_source(two_step_analysis)
```

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

The two helpers below format one of those results for display and are
specific to this document.

``` python
def as_table(result):
    """The per-coefficient results as a table, in the manuscript's format."""
    return pd.DataFrame({
        "Effect": COEF_NAMES,
        "Mean": [round(m, 2) for m in result["mean"]],
        "SD": [round(s, 2) for s in result["sd"]],
        "W": [round(w) for w in result["W"]],
        "P-value": [fmt_p(p) for p in result["p"]],
        "Effect size": [round(e, 3) for e in result["effect_size"]],
        "": ["" if i not in IMPOSSIBLE else "impossible" for i in range(8)],
    })


def describe_exclusions(result, drop_separated=False):
    parts = [f"N = {result['n']} fitted"]
    if result["n_separated"]:
        kept = "excluded" if drop_separated else "retained"
        parts.append(f"{result['n_separated']} quasi-separated ({kept})")
    if result["n_rank_deficient"]:
        parts.append(f"{result['n_rank_deficient']} rank deficient (excluded)")
    if result["n_failed"]:
        parts.append(f"{result['n_failed']} failed to fit (excluded)")
    return "; ".join(parts)
```

## The two-step analysis using the real future transition

Here <a href="#eq-full" class="quarto-xref">Equation 2</a> is fitted to
each participant’s real data by maximum likelihood, with the real future
transitions as the predictor $x_f$. This reproduces **Table 1** of the
manuscript.

``` python
real = two_step_analysis(tst, method="mle", future="real")
note_fits("Real future transition, MLE")
print(describe_exclusions(real))
as_table(real)
```

    N = 563 fitted; 229 quasi-separated (retained)

<div id="tbl-real">

Table 1

<div class="cell-output cell-output-display" data-execution_count="7">

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Effect | Mean | SD | W | P-value | Effect size |  |
|----|----|----|----|----|----|----|----|
| 0 | intercept | 2.63 | 2.98 | 2225 | \<0.001 | 0.972 |  |
| 1 | reward | 0.58 | 1.54 | 38278 | \<0.001 | 0.518 |  |
| 2 | previous transition | -0.32 | 1.33 | 70764 | 0.102 | -0.109 |  |
| 3 | reward:previous transition | 0.44 | 1.48 | 48165 | \<0.001 | 0.393 |  |
| 4 | future transition | -0.51 | 1.30 | 51266 | \<0.001 | -0.354 | impossible |
| 5 | reward:future transition | -0.07 | 1.18 | 75463 | 0.620 | -0.049 | impossible |
| 6 | previous transition:future transition | 0.19 | 1.22 | 73212 | 0.330 | 0.078 | impossible |
| 7 | reward:previous transition:future transition | -0.05 | 1.20 | 77343 | 0.620 | -0.026 | impossible |

</div>

</div>

</div>

``` python
spurious = [COEF_NAMES[i] for i in IMPOSSIBLE if real["p"][i] < 0.05]
print("Consistently nonzero although the true value is zero:")
for name in spurious:
    print(f"  {name}")
```

    Consistently nonzero although the true value is zero:
      future transition

Taken at face value, these results indicate that participants were more
likely to stay with their previous choice when the transition they were
about to experience was rare rather than common. Since that transition
had not occurred when the choice was made, the alternatives are
precognition or a defect in the analysis.

## The result does not depend on the predictor being real

Replacing the future transitions with pseudo-random numbers drawn from a
Bernoulli distribution with $p = 0.7$, matching the transition
probabilities of the task, gives the same result. This reproduces
**Table 2** of the manuscript.

``` python
simulated = two_step_analysis(tst, method="mle", future=P_COMMON,
                              check_sep=False)
note_fits("Simulated future transition (70/30), MLE")
print(describe_exclusions(simulated))
as_table(simulated)
```

    N = 562 fitted; 1 rank deficient (excluded)

<div id="tbl-simulated">

Table 2

<div class="cell-output cell-output-display" data-execution_count="9">

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Effect | Mean | SD | W | P-value | Effect size |  |
|----|----|----|----|----|----|----|----|
| 0 | intercept | 2.67 | 3.04 | 3120 | \<0.001 | 0.961 |  |
| 1 | reward | 0.54 | 1.51 | 39023 | \<0.001 | 0.507 |  |
| 2 | previous transition | -0.37 | 1.36 | 68366 | 0.021 | -0.136 |  |
| 3 | reward:previous transition | 0.44 | 1.58 | 50057 | \<0.001 | 0.367 |  |
| 4 | future transition | -0.53 | 1.32 | 56680 | \<0.001 | -0.283 | impossible |
| 5 | reward:future transition | -0.07 | 1.28 | 74770 | 0.522 | -0.055 | impossible |
| 6 | previous transition:future transition | 0.17 | 1.22 | 72517 | 0.262 | 0.083 | impossible |
| 7 | reward:previous transition:future transition | -0.12 | 1.28 | 76954 | 0.577 | -0.027 | impossible |

</div>

</div>

</div>

### Replications

``` python
def replicate(tstdf, method, future, drop_separated, n_reps=N_REPS,
              label=""):
    """Repeat an analysis with freshly generated future transitions and count
    how often at least one impossible coefficient comes out significant."""
    rows, hits = [], 0
    for rep in range(n_reps):
        res = two_step_analysis(tstdf, method=method, future=future,
                                drop_separated=drop_separated,
                                check_sep=drop_separated)
        hit = any(res["p"][i] < 0.05 for i in IMPOSSIBLE)
        hits += hit
        rows.append({
            "rep": rep + 1,
            "b_f": round(res["mean"][4], 3),
            "P (b_f)": fmt_p(res["p"][4]),
            "any impossible coef. significant": "yes" if hit else "no",
            "n": res["n"],
        })
    table = pd.DataFrame(rows)
    print(f"{label}\nAt least one impossible coefficient was significant in "
          f"{hits} of {n_reps} replications.")
    print(f"b_f ranged from {table.b_f.min():+.2f} to {table.b_f.max():+.2f}.")
    if drop_separated:
        n_part = tstdf.subj.nunique()
        print(f"Participants retained per replication: median "
              f"{table.n.median():.0f} of {n_part} "
              f"({n_part - table.n.median():.0f} quasi-separated, excluded).")
    else:
        excluded = (tstdf.subj.nunique() - table.n).max()
        if excluded:
            print(f"At most {excluded} participants were excluded from a "
                  "replication (rank deficient or failed to fit).")
    return table, hits


reps_mle, hits_mle = replicate(
    tst, "mle", P_COMMON, False,
    label="Simulated predictor (70/30), maximum likelihood")
note_fits(f"Simulated predictor, MLE, {N_REPS} replications")
reps_mle
```

    Simulated predictor (70/30), maximum likelihood
    At least one impossible coefficient was significant in 20 of 20 replications.
    b_f ranged from -0.63 to -0.47.
    At most 2 participants were excluded from a replication (rank deficient or failed to fit).

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | rep | b_f    | P (b_f) | any impossible coef. significant | n   |
|-----|-----|--------|---------|----------------------------------|-----|
| 0   | 1   | -0.485 | \<0.001 | yes                              | 563 |
| 1   | 2   | -0.560 | \<0.001 | yes                              | 563 |
| 2   | 3   | -0.497 | \<0.001 | yes                              | 563 |
| 3   | 4   | -0.550 | \<0.001 | yes                              | 563 |
| 4   | 5   | -0.632 | \<0.001 | yes                              | 562 |
| 5   | 6   | -0.575 | \<0.001 | yes                              | 563 |
| 6   | 7   | -0.573 | \<0.001 | yes                              | 563 |
| 7   | 8   | -0.513 | \<0.001 | yes                              | 563 |
| 8   | 9   | -0.518 | \<0.001 | yes                              | 563 |
| 9   | 10  | -0.505 | \<0.001 | yes                              | 563 |
| 10  | 11  | -0.529 | \<0.001 | yes                              | 563 |
| 11  | 12  | -0.477 | \<0.001 | yes                              | 563 |
| 12  | 13  | -0.518 | \<0.001 | yes                              | 563 |
| 13  | 14  | -0.498 | \<0.001 | yes                              | 561 |
| 14  | 15  | -0.572 | \<0.001 | yes                              | 562 |
| 15  | 16  | -0.584 | \<0.001 | yes                              | 563 |
| 16  | 17  | -0.468 | \<0.001 | yes                              | 563 |
| 17  | 18  | -0.518 | \<0.001 | yes                              | 563 |
| 18  | 19  | -0.548 | \<0.001 | yes                              | 563 |
| 19  | 20  | -0.568 | \<0.001 | yes                              | 563 |

</div>

Taken at face value, participants anticipated not only a future task
event but also future computer simulations.

## What creates the effect: the unequal split

The two values of $x_f$ are not equally frequent: it is $+1$ on 70% of
trial pairs and $-1$ on the remaining 30%. Generating the same predictor
with a 50% probability of $+1$, so that the two subsets of trial pairs
are the same size, removes the effect entirely.

``` python
reps_balanced, hits_balanced = replicate(
    tst, "mle", 0.5, False,
    label="Balanced predictor (50/50), maximum likelihood "
          "(expected: about 1 replication in 20)")
note_fits(f"Balanced predictor, MLE, {N_REPS} replications")
reps_balanced
```

    Balanced predictor (50/50), maximum likelihood (expected: about 1 replication in 20)
    At least one impossible coefficient was significant in 0 of 20 replications.
    b_f ranged from -0.12 to +0.08.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | rep | b_f    | P (b_f) | any impossible coef. significant | n   |
|-----|-----|--------|---------|----------------------------------|-----|
| 0   | 1   | -0.097 | 1.000   | no                               | 563 |
| 1   | 2   | -0.002 | 1.000   | no                               | 563 |
| 2   | 3   | 0.005  | 1.000   | no                               | 563 |
| 3   | 4   | -0.015 | 1.000   | no                               | 563 |
| 4   | 5   | 0.036  | 1.000   | no                               | 563 |
| 5   | 6   | 0.077  | 0.936   | no                               | 563 |
| 6   | 7   | 0.004  | 1.000   | no                               | 563 |
| 7   | 8   | 0.020  | 0.221   | no                               | 563 |
| 8   | 9   | -0.117 | 0.130   | no                               | 563 |
| 9   | 10  | -0.022 | 1.000   | no                               | 563 |
| 10  | 11  | -0.084 | 0.196   | no                               | 563 |
| 11  | 12  | -0.041 | 1.000   | no                               | 563 |
| 12  | 13  | -0.025 | 1.000   | no                               | 563 |
| 13  | 14  | 0.043  | 1.000   | no                               | 563 |
| 14  | 15  | -0.063 | 0.788   | no                               | 563 |
| 15  | 16  | 0.046  | 1.000   | no                               | 563 |
| 16  | 17  | -0.020 | 1.000   | no                               | 563 |
| 17  | 18  | 0.002  | 1.000   | no                               | 563 |
| 18  | 19  | 0.005  | 0.975   | no                               | 563 |
| 19  | 20  | -0.014 | 1.000   | no                               | 563 |

</div>

The bias in each subset remains, since each is still estimated from
approximately a hundred trial pairs, but the two biases now have the
same expected magnitude and cancel when the difference between them is
taken.

### Why the difference between the two biases is what the coefficient represents

Because the future transition is entered together with all of its
interactions, the eight-term model
(<a href="#eq-full" class="quarto-xref">Equation 2</a>) is equivalent to
fitting the original four-term model
(<a href="#eq-usual" class="quarto-xref">Equation 1</a>) twice: once to
the subset of trial pairs in which the upcoming transition is common,
and once to the subset in which it is rare. Each of the first four
coefficients of the eight-term model is then the average of its two
subset values, and each of the four coefficients involving the future
transition is half the difference between them.

Suppose a participant’s true intercept is 1. Both subsets tend to
overestimate it, since maximum likelihood inflates the magnitude of a
logistic regression coefficient in small samples, but the rare subset,
having approximately 60 trial pairs against 140, tends to be more
inflated: say 1.2 in the common subset and 1.6 in the rare. The fitted
intercept is then $\tfrac{1}{2}(1.2 + 1.6) = 1.4$, and the coefficient
on the future transition is $\tfrac{1}{2}(1.2 - 1.6) = -0.2$. The future
transition has no effect whatsoever, and the coefficient is negative
purely because the estimate it is derived from was more strongly
inflated on one side than the other.

The sign of each future-transition coefficient should therefore be
opposite to the sign of the term it derives from.

``` python
pos_intercept = (real["coefs"][:, 0] > 0).mean()
print(f"Intercept estimated as positive for {pos_intercept:.0%} of participants")

pd.DataFrame({
    "Term": COEF_NAMES[:4],
    "Mean coefficient": [round(real["mean"][i], 2) for i in range(4)],
    "Future-transition counterpart": COEF_NAMES[4:],
    "Mean counterpart": [round(real["mean"][i], 2) for i in range(4, 8)],
})
```

    Intercept estimated as positive for 94% of participants

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Term | Mean coefficient | Future-transition counterpart | Mean counterpart |
|----|----|----|----|----|
| 0 | intercept | 2.63 | future transition | -0.51 |
| 1 | reward | 0.58 | reward:future transition | -0.07 |
| 2 | previous transition | -0.32 | previous transition:future transition | 0.19 |
| 3 | reward:previous transition | 0.44 | reward:previous transition:future transition | -0.05 |

</div>

### A simulation confirms the account

Data are generated from a model with no future-transition effect at all
and an intercept fixed at values from $-3$ to $+3$; the eight-term model
is then fitted to each simulated participant and the coefficients tested
across the sample as before. This reproduces **Table 3** of the
manuscript.

``` python
def simulate_intercepts(intercepts, n_part=500, n_trials=200,
                        p_common=P_COMMON, p_reward=0.56):
    """Generate stay/switch data from a known model with no future-transition
    effect, fit it the same way, and record the future-transition coefficient
    for each value of the intercept.  The other coefficients are the means of
    the corresponding estimates in the real-data analysis, and p_reward
    matches the marginal frequency of reward in the dataset."""
    rows = []
    for b0 in intercepts:
        coefs = []
        for _ in range(n_part):
            rw = 2 * (rng.random(n_trials) < p_reward) - 1
            tr = 2 * (rng.random(n_trials) < p_common) - 1
            ft = 2 * (rng.random(n_trials) < p_common) - 1
            # the future transition is absent from the generating model
            p = expit(b0 + 0.58 * rw - 0.32 * tr + 0.44 * rw * tr)
            y = (rng.random(n_trials) < p).astype(int)
            X = np.column_stack([np.ones(n_trials), rw, tr, rw * tr,
                                 ft, ft * rw, ft * tr, ft * rw * tr])
            if y.min() == y.max():
                continue
            params, status, _ = participant_status(y, X, method="mle",
                                                   check_sep=False)
            if status == "ok":
                coefs.append(params)
        g = np.array([c[4] for c in coefs])
        rows.append({
            "True intercept": b0,
            "b_f estimate": round(g.mean(), 3),
            "P-value": fmt_p(stats.wilcoxon(g).pvalue),
            "N": len(g),
        })
    return pd.DataFrame(rows)


intercept_sim = simulate_intercepts([-3, -2, -1, 0, 1, 2, 3])
note_fits("Intercept simulation (7 x 500 simulated participants)")
intercept_sim
```

<div id="tbl-intercepts">

Table 3

<div class="cell-output cell-output-display" data-execution_count="13">

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | True intercept | b_f estimate | P-value | N   |
|-----|----------------|--------------|---------|-----|
| 0   | -3             | 1.261        | \<0.001 | 500 |
| 1   | -2             | 1.078        | \<0.001 | 500 |
| 2   | -1             | 0.408        | \<0.001 | 500 |
| 3   | 0              | -0.014       | 0.298   | 500 |
| 4   | 1              | -0.534       | \<0.001 | 500 |
| 5   | 2              | -1.263       | \<0.001 | 500 |
| 6   | 3              | -1.274       | \<0.001 | 500 |

</div>

</div>

</div>

The estimated future-transition coefficient takes the opposite sign to
the intercept at every value, grows with the magnitude of the intercept
over most of the range, and is indistinguishable from zero when the
intercept is zero.

## The effect is not caused by the degenerate participants

In a design with eight terms, each participant’s 200 trial pairs are
divided among eight subsets defined by the reward and the two
transitions, and the sparsest of these contains fewer than ten pairs on
average. Where such a subset contains no variation in the outcome, the
maximum likelihood estimate does not exist: the likelihood increases
without bound as the coefficients grow, and the value an optimiser
returns reflects its convergence tolerance rather than the participant’s
behaviour. The linear-programming test of Konis (2007) detects this
condition, known as quasi-complete separation.

``` python
p_rewarded = ((tst.rw == 1).mean())
print(f"Frequency of rewarded trials: {p_rewarded:.2f}; "
      f"unrewarded: {1 - p_rewarded:.2f}")
print("Expected size of the sparsest design cell: "
      f"200 x {1 - p_rewarded:.2f} x 0.3 x 0.3 = "
      f"{200 * (1 - p_rewarded) * 0.3 * 0.3:.0f} trial pairs")
print()
print(f"Quasi-complete separation: {real['n_separated']} of {real['n']} "
      "participants")
print("Every one of those fits was reported by the optimiser as converged.")
```

    Frequency of rewarded trials: 0.56; unrewarded: 0.44
    Expected size of the sparsest design cell: 200 x 0.44 x 0.3 x 0.3 = 8 trial pairs

    Quasi-complete separation: 229 of 563 participants
    Every one of those fits was reported by the optimiser as converged.

Excluding these participants, however, does not help.
<a href="#sec-firth" class="quarto-xref">Section 9</a> repeats the
simulated-predictor analysis with Firth’s penalised likelihood, once
with every participant and once discarding those whose estimates are
formally infinite.

## Bias reduction reduces the artefact but does not remove it

Firth’s penalised likelihood removes the leading $O(n^{-1})$ term of the
bias of the maximum likelihood estimator and yields finite estimates
under separation. Refitting each participant with it reproduces **Table
4** of the manuscript.

``` python
firth = two_step_analysis(tst, method="firth", future="real", check_sep=False)
note_fits("Real future transition, Firth")
print(describe_exclusions(firth))
as_table(firth)
```

    N = 563 fitted

<div id="tbl-firth">

Table 4

<div class="cell-output cell-output-display" data-execution_count="15">

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Effect | Mean | SD | W | P-value | Effect size |  |
|----|----|----|----|----|----|----|----|
| 0 | intercept | 1.23 | 0.91 | 1626 | \<0.001 | 0.980 |  |
| 1 | reward | 0.27 | 0.32 | 16798 | \<0.001 | 0.788 |  |
| 2 | previous transition | 0.09 | 0.24 | 47425 | \<0.001 | 0.403 |  |
| 3 | reward:previous transition | 0.20 | 0.33 | 31107 | \<0.001 | 0.608 |  |
| 4 | future transition | 0.04 | 0.23 | 62519 | \<0.001 | 0.212 | impossible |
| 5 | reward:future transition | 0.01 | 0.20 | 75442 | 0.308 | 0.050 | impossible |
| 6 | previous transition:future transition | -0.02 | 0.21 | 71382 | 0.115 | -0.101 | impossible |
| 7 | reward:previous transition:future transition | 0.02 | 0.21 | 72071 | 0.117 | 0.092 | impossible |

</div>

</div>

</div>

``` python
pd.DataFrame({
    "Effect": [COEF_NAMES[i] for i in IMPOSSIBLE],
    "MLE mean": [round(real["mean"][i], 3) for i in IMPOSSIBLE],
    "Firth mean": [round(firth["mean"][i], 3) for i in IMPOSSIBLE],
    "MLE P": [fmt_p(real["p"][i]) for i in IMPOSSIBLE],
    "Firth P": [fmt_p(firth["p"][i]) for i in IMPOSSIBLE],
})
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Effect | MLE mean | Firth mean | MLE P | Firth P |
|----|----|----|----|----|----|
| 0 | future transition | -0.509 | 0.044 | \<0.001 | \<0.001 |
| 1 | reward:future transition | -0.066 | 0.008 | 0.620 | 0.308 |
| 2 | previous transition:future transition | 0.187 | -0.020 | 0.330 | 0.115 |
| 3 | reward:previous transition:future transition | -0.048 | 0.021 | 0.620 | 0.117 |

</div>

The coefficient on the future transition falls by roughly an order of
magnitude. It nevertheless remains detectable across replications,
because Firth’s method leaves a remainder of order $n^{-2}$, and with
563 participants, a shared bias of that size is still distinguishable
from zero.

``` python
reps_firth, hits_firth = replicate(
    tst, "firth", P_COMMON, False,
    label="Simulated predictor (70/30), Firth penalised likelihood")
note_fits(f"Simulated predictor, Firth, {N_REPS} replications")
reps_firth
```

    Simulated predictor (70/30), Firth penalised likelihood
    At least one impossible coefficient was significant in 18 of 20 replications.
    b_f ranged from +0.02 to +0.05.
    At most 2 participants were excluded from a replication (rank deficient or failed to fit).

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | rep | b_f   | P (b_f) | any impossible coef. significant | n   |
|-----|-----|-------|---------|----------------------------------|-----|
| 0   | 1   | 0.037 | 0.001   | yes                              | 563 |
| 1   | 2   | 0.018 | 0.268   | yes                              | 563 |
| 2   | 3   | 0.036 | 0.001   | yes                              | 562 |
| 3   | 4   | 0.019 | 0.166   | no                               | 563 |
| 4   | 5   | 0.039 | 0.001   | yes                              | 562 |
| 5   | 6   | 0.036 | 0.003   | yes                              | 563 |
| 6   | 7   | 0.044 | \<0.001 | yes                              | 563 |
| 7   | 8   | 0.031 | 0.045   | yes                              | 563 |
| 8   | 9   | 0.030 | 0.029   | yes                              | 563 |
| 9   | 10  | 0.037 | 0.003   | yes                              | 562 |
| 10  | 11  | 0.021 | 0.150   | no                               | 561 |
| 11  | 12  | 0.042 | \<0.001 | yes                              | 563 |
| 12  | 13  | 0.023 | 0.035   | yes                              | 563 |
| 13  | 14  | 0.040 | \<0.001 | yes                              | 563 |
| 14  | 15  | 0.040 | \<0.001 | yes                              | 563 |
| 15  | 16  | 0.047 | \<0.001 | yes                              | 563 |
| 16  | 17  | 0.031 | 0.011   | yes                              | 563 |
| 17  | 18  | 0.037 | 0.001   | yes                              | 562 |
| 18  | 19  | 0.045 | \<0.001 | yes                              | 563 |
| 19  | 20  | 0.028 | 0.021   | yes                              | 563 |

</div>

``` python
reps_firth_dropped, hits_firth_dropped = replicate(
    tst, "firth", P_COMMON, True,
    label="Simulated predictor (70/30), Firth, separated participants excluded")
note_fits(f"Simulated predictor, Firth, separated excluded, {N_REPS} replications")
reps_firth_dropped
```

    Simulated predictor (70/30), Firth, separated participants excluded
    At least one impossible coefficient was significant in 20 of 20 replications.
    b_f ranged from +0.03 to +0.06.
    Participants retained per replication: median 332 of 563 (232 quasi-separated, excluded).

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | rep | b_f   | P (b_f) | any impossible coef. significant | n   |
|-----|-----|-------|---------|----------------------------------|-----|
| 0   | 1   | 0.041 | \<0.001 | yes                              | 331 |
| 1   | 2   | 0.051 | \<0.001 | yes                              | 323 |
| 2   | 3   | 0.032 | 0.010   | yes                              | 338 |
| 3   | 4   | 0.033 | 0.014   | yes                              | 340 |
| 4   | 5   | 0.055 | \<0.001 | yes                              | 317 |
| 5   | 6   | 0.056 | \<0.001 | yes                              | 312 |
| 6   | 7   | 0.039 | \<0.001 | yes                              | 339 |
| 7   | 8   | 0.037 | 0.003   | yes                              | 346 |
| 8   | 9   | 0.031 | 0.016   | yes                              | 333 |
| 9   | 10  | 0.026 | 0.026   | yes                              | 323 |
| 10  | 11  | 0.054 | \<0.001 | yes                              | 339 |
| 11  | 12  | 0.057 | \<0.001 | yes                              | 328 |
| 12  | 13  | 0.049 | \<0.001 | yes                              | 335 |
| 13  | 14  | 0.038 | 0.004   | yes                              | 325 |
| 14  | 15  | 0.030 | 0.005   | yes                              | 333 |
| 15  | 16  | 0.051 | \<0.001 | yes                              | 322 |
| 16  | 17  | 0.039 | 0.002   | yes                              | 316 |
| 17  | 18  | 0.040 | 0.003   | yes                              | 332 |
| 18  | 19  | 0.044 | \<0.001 | yes                              | 333 |
| 19  | 20  | 0.029 | 0.021   | yes                              | 325 |

</div>

``` python
print(f"Mean b_f, all participants:        {reps_firth.b_f.mean():+.3f}")
print(f"Mean b_f, separated excluded:      {reps_firth_dropped.b_f.mean():+.3f}")
print(f"Significant in {hits_firth} of {N_REPS} and "
      f"{hits_firth_dropped} of {N_REPS} replications respectively.")
```

    Mean b_f, all participants:        +0.034
    Mean b_f, separated excluded:      +0.042
    Significant in 18 of 20 and 20 of 20 replications respectively.

The artefact is not produced by the participants whose data are too
sparse to support the model. It is produced by unequal information
across the two subsets, which is a property of every participant’s data
alike.

## How much data constrain each coefficient

Two hundred trial pairs for eight parameters may not seem obviously
insufficient, but the relevant number for a logistic model is the number
of *events*, that is, the rarer of the two binary outcomes. Peduzzi et
al. (1996) recommend at least ten events per parameter.

``` python
rows = []
for subj, partdf in tst.groupby("subj"):
    y, _ = build_design(partdf)
    n_pairs = len(y)
    n_stay = int(y.sum())
    rows.append({
        "subj": subj,
        "n_pairs": n_pairs,
        "stay_prop": n_stay / n_pairs,
        "events": min(n_stay, n_pairs - n_stay),
    })
epv = pd.DataFrame(rows)
epv["epv_8term"] = epv.events / 8

print(f"Total trial pairs analysed: {epv.n_pairs.sum():,} "
      f"(for {len(epv)} participants)")
print(f"Median stay proportion: {epv.stay_prop.median():.0%}")
print(f"Median events per participant: {epv.events.median():.0f}")
print(f"Median events per parameter (8-term model): "
      f"{epv.epv_8term.median():.1f}")
print()
print(f"Below 10 events per parameter: {(epv.epv_8term < 10).mean():.0%}")
print(f"Below  5 events per parameter: {(epv.epv_8term < 5).mean():.0%}")
print(f"Below  2 events per parameter: {(epv.epv_8term < 2).mean():.0%}")
print()
print("Four-term model in standard use (requires 40 events):")
print(f"  falling short: {(epv.events < 40).mean():.0%} of participants")
```

    Total trial pairs analysed: 110,656 (for 563 participants)
    Median stay proportion: 76%
    Median events per participant: 47
    Median events per parameter (8-term model): 5.9

    Below 10 events per parameter: 82%
    Below  5 events per parameter: 42%
    Below  2 events per parameter: 12%

    Four-term model in standard use (requires 40 events):
      falling short: 42% of participants

The eight-term model used here naturally does worse than the four-term
model in standard use, but that model also fails the criterion for a
large fraction of participants.

## A pooled fit as a quick check

Where a parameter is tested against a fixed value rather than compared
across groups, a cheap check is available that does not require the
analysis to be redone. Fit the same model once to all the trial pairs
pooled across participants, ignoring the division into participants
entirely, and compare the result with the average of the
participant-level estimates. Fitting the model simultaneously to the
combined data reduces the sample-size-dependent bias by a large factor.

Trial pairs are still built within each participant, so that no pair
spans two participants.

``` python
ys, Xs = [], []
for _, partdf in tst.groupby("subj"):
    y, X = build_design(partdf, future="real")
    ys.append(y)
    Xs.append(X)
y_pooled = np.concatenate(ys)
X_pooled = np.vstack(Xs)

pooled = sm.Logit(y_pooled, X_pooled).fit(method="bfgs", maxiter=1000, disp=0)
assert pooled.mle_retvals["converged"]
FITS["total"] += 1          # this fit does not go through fit_participant
note_fits("Pooled fit across all participants")

pd.DataFrame({
    "Effect": COEF_NAMES,
    "Pooled estimate": np.round(pooled.params, 3),
    "Pooled P": [fmt_p(p) for p in pooled.pvalues],
    "Mean of participant-level estimates": [round(m, 2) for m in real["mean"]],
    "Two-step P": [fmt_p(p) for p in real["p"]],
})
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Effect | Pooled estimate | Pooled P | Mean of participant-level estimates | Two-step P |
|----|----|----|----|----|----|
| 0 | intercept | 1.049 | \<0.001 | 2.63 | \<0.001 |
| 1 | reward | 0.203 | \<0.001 | 0.58 | \<0.001 |
| 2 | previous transition | 0.041 | \<0.001 | -0.32 | 0.102 |
| 3 | reward:previous transition | 0.159 | \<0.001 | 0.44 | \<0.001 |
| 4 | future transition | 0.002 | 0.843 | -0.51 | \<0.001 |
| 5 | reward:future transition | 0.000 | 0.972 | -0.07 | 0.620 |
| 6 | previous transition:future transition | -0.012 | 0.130 | 0.19 | 0.330 |
| 7 | reward:previous transition:future transition | 0.007 | 0.382 | -0.05 | 0.620 |

</div>

## How many model fits are performed for this document

The analyses above are the same set the original script performed, and
the fit counts should therefore match it. With `N_REPS = 20` the total
is close to 45,571; it is not fixed exactly, because the number of
quasi-separated participants varies from replication to replication and
a few simulated participants in the intercept simulation produce no
variation in the outcome and are skipped before any model is fitted.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | Analysis                                          | Model fits |
|-----|---------------------------------------------------|------------|
| 0   | Real future transition, MLE                       | 563        |
| 1   | Simulated future transition (70/30), MLE          | 562        |
| 2   | Simulated predictor, MLE, 20 replications         | 11256      |
| 3   | Balanced predictor, MLE, 20 replications          | 11260      |
| 4   | Intercept simulation (7 x 500 simulated partic... | 3500       |
| 5   | Real future transition, Firth                     | 563        |
| 6   | Simulated predictor, Firth, 20 replications       | 11254      |
| 7   | Simulated predictor, Firth, separated excluded... | 6590       |
| 8   | Pooled fit across all participants                | 1          |
| 9   | Total                                             | 45549      |

</div>

## References

- Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P., & Dolan, R. J.
  (2011). Model-based influences on humans’ choices and striatal
  prediction errors. *Neuron*, 69(6), 1204–1215.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates.
  *Biometrika*, 80(1), 27–38.
- Konis, K. (2007). Linear programming algorithms for detection of
  separated data in binary logistic regression models. DPhil thesis,
  University of Oxford.
- Peduzzi, P., Concato, J., Kemper, E., Holford, T. R., &
  Feinstein, A. R. (1996). A simulation study of the number of events
  per variable in logistic regression analysis. *Journal of Clinical
  Epidemiology*, 49(12), 1373–1379.
- Shahar, N., Moran, R., Hauser, T. U., Kievit, R. A., McNamee, D.,
  Moutoussis, M., NSPN Consortium, Dolan, R. J., Bullmore, E., Dolan,
  R., Goodyer, I., Fonagy, P., Jones, P., Moutoussis, M., Hauser, T.,
  Neufeld, S., Romero-Garcia, R., St Clair, M., Vértes, P., . . .
  Kievit, R. (2019). Credit assignment to state-independent task
  representations and its relationship with model-based decision making.
  *Proceedings of the National Academy of Sciences*, 116(32),
  15871–15876. <https://doi.org/10.1073/pnas.1821647116> Data:
  <https://osf.io/7dekj/>
