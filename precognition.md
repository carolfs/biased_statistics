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
than 568 × 200.

> **Columns used**
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
choice, and regresses it on the first trial’s outcome *x*<sub>*r*</sub>
(+1 rewarded, −1 unrewarded) and transition type *x*<sub>*t*</sub> (+1
common, −1 rare):

<span id="eq-usual">
logit(*P*(stay)) = *β*<sub>0</sub> + *β*<sub>*r*</sub>*x*<sub>*r*</sub> + *β*<sub>*t*</sub>*x*<sub>*t*</sub> + *β*<sub>*r* × *t*</sub>*x*<sub>*r*</sub>*x*<sub>*t*</sub>.   (1)
</span>

To this model, I add the *second* trial’s transition, *x*<sub>*f*</sub>,
where *f* stands for “future”, together with all of its interactions:

<span id="eq-full">
logit(*P*(stay)) = *β*<sub>0</sub> + *β*<sub>*r*</sub>*x*<sub>*r*</sub> + *β*<sub>*t*</sub>*x*<sub>*t*</sub> + *β*<sub>*r* × *t*</sub>*x*<sub>*r*</sub>*x*<sub>*t*</sub> + *β*<sub>*f*</sub>*x*<sub>*f*</sub> + *β*<sub>*r* × *f*</sub>*x*<sub>*r*</sub>*x*<sub>*f*</sub> + *β*<sub>*t* × *f*</sub>*x*<sub>*t*</sub>*x*<sub>*f*</sub> + *β*<sub>*r* × *t* × *f*</sub>*x*<sub>*r*</sub>*x*<sub>*t*</sub>*x*<sub>*f*</sub>.   (2)
</span>

The stay-or-switch decision in each pair is made *before* the second
trial’s transition occurs, and each transition is an independent random
event with fixed probabilities. Every coefficient involving
*x*<sub>*f*</sub> therefore has a true value of zero.

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
combinations of *x*<sub>*r*</sub>, *x*<sub>*t*</sub> and
*x*<sub>*f*</sub> is absent from that participant’s trial pairs, in
which case the model is not identified, nothing is fitted, and the
separation test cannot be run either.

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
signed-rank test, with *P* values corrected for multiple comparisons by
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
transitions as the predictor *x*<sub>*f*</sub>. This reproduces **Table
1** of the manuscript.

``` python
real = two_step_analysis(tst, method="mle", future="real")
note_fits("Real future transition, MLE")
print(describe_exclusions(real))
as_table(real)
```

    N = 563 fitted; 229 quasi-separated (retained)

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">Effect</th>
<th data-quarto-table-cell-role="th">Mean</th>
<th data-quarto-table-cell-role="th">SD</th>
<th data-quarto-table-cell-role="th">W</th>
<th data-quarto-table-cell-role="th">P-value</th>
<th data-quarto-table-cell-role="th">Effect size</th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>intercept</td>
<td>2.63</td>
<td>2.98</td>
<td>2225</td>
<td>&lt;0.001</td>
<td>0.972</td>
<td></td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>reward</td>
<td>0.58</td>
<td>1.54</td>
<td>38278</td>
<td>&lt;0.001</td>
<td>0.518</td>
<td></td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>previous transition</td>
<td>-0.32</td>
<td>1.33</td>
<td>70764</td>
<td>0.102</td>
<td>-0.109</td>
<td></td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>reward:previous transition</td>
<td>0.44</td>
<td>1.48</td>
<td>48165</td>
<td>&lt;0.001</td>
<td>0.393</td>
<td></td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>future transition</td>
<td>-0.51</td>
<td>1.30</td>
<td>51266</td>
<td>&lt;0.001</td>
<td>-0.354</td>
<td>impossible</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5</td>
<td>reward:future transition</td>
<td>-0.07</td>
<td>1.18</td>
<td>75463</td>
<td>0.620</td>
<td>-0.049</td>
<td>impossible</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">6</td>
<td>previous transition:future transition</td>
<td>0.19</td>
<td>1.22</td>
<td>73212</td>
<td>0.330</td>
<td>0.078</td>
<td>impossible</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">7</td>
<td>reward:previous transition:future transition</td>
<td>-0.05</td>
<td>1.20</td>
<td>77343</td>
<td>0.620</td>
<td>-0.026</td>
<td>impossible</td>
</tr>
</tbody>
</table>

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
Bernoulli distribution with *p* = 0.7, matching the transition
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

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">Effect</th>
<th data-quarto-table-cell-role="th">Mean</th>
<th data-quarto-table-cell-role="th">SD</th>
<th data-quarto-table-cell-role="th">W</th>
<th data-quarto-table-cell-role="th">P-value</th>
<th data-quarto-table-cell-role="th">Effect size</th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>intercept</td>
<td>2.67</td>
<td>3.04</td>
<td>3120</td>
<td>&lt;0.001</td>
<td>0.961</td>
<td></td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>reward</td>
<td>0.54</td>
<td>1.51</td>
<td>39023</td>
<td>&lt;0.001</td>
<td>0.507</td>
<td></td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>previous transition</td>
<td>-0.37</td>
<td>1.36</td>
<td>68366</td>
<td>0.021</td>
<td>-0.136</td>
<td></td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>reward:previous transition</td>
<td>0.44</td>
<td>1.58</td>
<td>50057</td>
<td>&lt;0.001</td>
<td>0.367</td>
<td></td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>future transition</td>
<td>-0.53</td>
<td>1.32</td>
<td>56680</td>
<td>&lt;0.001</td>
<td>-0.283</td>
<td>impossible</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5</td>
<td>reward:future transition</td>
<td>-0.07</td>
<td>1.28</td>
<td>74770</td>
<td>0.522</td>
<td>-0.055</td>
<td>impossible</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">6</td>
<td>previous transition:future transition</td>
<td>0.17</td>
<td>1.22</td>
<td>72517</td>
<td>0.262</td>
<td>0.083</td>
<td>impossible</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">7</td>
<td>reward:previous transition:future transition</td>
<td>-0.12</td>
<td>1.28</td>
<td>76954</td>
<td>0.577</td>
<td>-0.027</td>
<td>impossible</td>
</tr>
</tbody>
</table>

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

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">rep</th>
<th data-quarto-table-cell-role="th">b_f</th>
<th data-quarto-table-cell-role="th">P (b_f)</th>
<th data-quarto-table-cell-role="th">any impossible coef.
significant</th>
<th data-quarto-table-cell-role="th">n</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>1</td>
<td>-0.485</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>2</td>
<td>-0.560</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>3</td>
<td>-0.497</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>4</td>
<td>-0.550</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>5</td>
<td>-0.632</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>562</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5</td>
<td>6</td>
<td>-0.575</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">6</td>
<td>7</td>
<td>-0.573</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">7</td>
<td>8</td>
<td>-0.513</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">8</td>
<td>9</td>
<td>-0.518</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">9</td>
<td>10</td>
<td>-0.505</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">10</td>
<td>11</td>
<td>-0.529</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">11</td>
<td>12</td>
<td>-0.477</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">12</td>
<td>13</td>
<td>-0.518</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">13</td>
<td>14</td>
<td>-0.498</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>561</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">14</td>
<td>15</td>
<td>-0.572</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>562</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">15</td>
<td>16</td>
<td>-0.584</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">16</td>
<td>17</td>
<td>-0.468</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">17</td>
<td>18</td>
<td>-0.518</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">18</td>
<td>19</td>
<td>-0.548</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">19</td>
<td>20</td>
<td>-0.568</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
</tbody>
</table>

</div>

Taken at face value, participants anticipated not only a future task
event but also future computer simulations.

## What creates the effect: the unequal split

The two values of *x*<sub>*f*</sub> are not equally frequent: it is +1
on 70% of trial pairs and −1 on the remaining 30%. Generating the same
predictor with a 50% probability of +1, so that the two subsets of trial
pairs are the same size, removes the effect entirely.

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

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">rep</th>
<th data-quarto-table-cell-role="th">b_f</th>
<th data-quarto-table-cell-role="th">P (b_f)</th>
<th data-quarto-table-cell-role="th">any impossible coef.
significant</th>
<th data-quarto-table-cell-role="th">n</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>1</td>
<td>-0.097</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>2</td>
<td>-0.002</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>3</td>
<td>0.005</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>4</td>
<td>-0.015</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>5</td>
<td>0.036</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5</td>
<td>6</td>
<td>0.077</td>
<td>0.936</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">6</td>
<td>7</td>
<td>0.004</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">7</td>
<td>8</td>
<td>0.020</td>
<td>0.221</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">8</td>
<td>9</td>
<td>-0.117</td>
<td>0.130</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">9</td>
<td>10</td>
<td>-0.022</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">10</td>
<td>11</td>
<td>-0.084</td>
<td>0.196</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">11</td>
<td>12</td>
<td>-0.041</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">12</td>
<td>13</td>
<td>-0.025</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">13</td>
<td>14</td>
<td>0.043</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">14</td>
<td>15</td>
<td>-0.063</td>
<td>0.788</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">15</td>
<td>16</td>
<td>0.046</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">16</td>
<td>17</td>
<td>-0.020</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">17</td>
<td>18</td>
<td>0.002</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">18</td>
<td>19</td>
<td>0.005</td>
<td>0.975</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">19</td>
<td>20</td>
<td>-0.014</td>
<td>1.000</td>
<td>no</td>
<td>563</td>
</tr>
</tbody>
</table>

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

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">Term</th>
<th data-quarto-table-cell-role="th">Mean coefficient</th>
<th data-quarto-table-cell-role="th">Future-transition counterpart</th>
<th data-quarto-table-cell-role="th">Mean counterpart</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>intercept</td>
<td>2.63</td>
<td>future transition</td>
<td>-0.51</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>reward</td>
<td>0.58</td>
<td>reward:future transition</td>
<td>-0.07</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>previous transition</td>
<td>-0.32</td>
<td>previous transition:future transition</td>
<td>0.19</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>reward:previous transition</td>
<td>0.44</td>
<td>reward:previous transition:future transition</td>
<td>-0.05</td>
</tr>
</tbody>
</table>

</div>

### A simulation confirms the account

Data are generated from a model with no future-transition effect at all
and an intercept fixed at values from −3 to +3; the eight-term model is
then fitted to each simulated participant and the coefficients tested
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

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">True intercept</th>
<th data-quarto-table-cell-role="th">b_f estimate</th>
<th data-quarto-table-cell-role="th">P-value</th>
<th data-quarto-table-cell-role="th">N</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>-3</td>
<td>1.261</td>
<td>&lt;0.001</td>
<td>500</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>-2</td>
<td>1.078</td>
<td>&lt;0.001</td>
<td>500</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>-1</td>
<td>0.408</td>
<td>&lt;0.001</td>
<td>500</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>0</td>
<td>-0.014</td>
<td>0.298</td>
<td>500</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>1</td>
<td>-0.534</td>
<td>&lt;0.001</td>
<td>500</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5</td>
<td>2</td>
<td>-1.263</td>
<td>&lt;0.001</td>
<td>500</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">6</td>
<td>3</td>
<td>-1.274</td>
<td>&lt;0.001</td>
<td>500</td>
</tr>
</tbody>
</table>

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

Firth’s penalised likelihood removes the leading *O*(*n*<sup>−1</sup>)
term of the bias of the maximum likelihood estimator and yields finite
estimates under separation. Refitting each participant with it
reproduces **Table 4** of the manuscript.

``` python
firth = two_step_analysis(tst, method="firth", future="real", check_sep=False)
note_fits("Real future transition, Firth")
print(describe_exclusions(firth))
as_table(firth)
```

    N = 563 fitted

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">Effect</th>
<th data-quarto-table-cell-role="th">Mean</th>
<th data-quarto-table-cell-role="th">SD</th>
<th data-quarto-table-cell-role="th">W</th>
<th data-quarto-table-cell-role="th">P-value</th>
<th data-quarto-table-cell-role="th">Effect size</th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>intercept</td>
<td>1.23</td>
<td>0.91</td>
<td>1626</td>
<td>&lt;0.001</td>
<td>0.980</td>
<td></td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>reward</td>
<td>0.27</td>
<td>0.32</td>
<td>16798</td>
<td>&lt;0.001</td>
<td>0.788</td>
<td></td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>previous transition</td>
<td>0.09</td>
<td>0.24</td>
<td>47425</td>
<td>&lt;0.001</td>
<td>0.403</td>
<td></td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>reward:previous transition</td>
<td>0.20</td>
<td>0.33</td>
<td>31107</td>
<td>&lt;0.001</td>
<td>0.608</td>
<td></td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>future transition</td>
<td>0.04</td>
<td>0.23</td>
<td>62519</td>
<td>&lt;0.001</td>
<td>0.212</td>
<td>impossible</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5</td>
<td>reward:future transition</td>
<td>0.01</td>
<td>0.20</td>
<td>75442</td>
<td>0.308</td>
<td>0.050</td>
<td>impossible</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">6</td>
<td>previous transition:future transition</td>
<td>-0.02</td>
<td>0.21</td>
<td>71382</td>
<td>0.115</td>
<td>-0.101</td>
<td>impossible</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">7</td>
<td>reward:previous transition:future transition</td>
<td>0.02</td>
<td>0.21</td>
<td>72071</td>
<td>0.117</td>
<td>0.092</td>
<td>impossible</td>
</tr>
</tbody>
</table>

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

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">Effect</th>
<th data-quarto-table-cell-role="th">MLE mean</th>
<th data-quarto-table-cell-role="th">Firth mean</th>
<th data-quarto-table-cell-role="th">MLE P</th>
<th data-quarto-table-cell-role="th">Firth P</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>future transition</td>
<td>-0.509</td>
<td>0.044</td>
<td>&lt;0.001</td>
<td>&lt;0.001</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>reward:future transition</td>
<td>-0.066</td>
<td>0.008</td>
<td>0.620</td>
<td>0.308</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>previous transition:future transition</td>
<td>0.187</td>
<td>-0.020</td>
<td>0.330</td>
<td>0.115</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>reward:previous transition:future transition</td>
<td>-0.048</td>
<td>0.021</td>
<td>0.620</td>
<td>0.117</td>
</tr>
</tbody>
</table>

</div>

The coefficient on the future transition falls by roughly an order of
magnitude. It nevertheless remains detectable across replications,
because Firth’s method leaves a remainder of order *n*<sup>−2</sup>, and
with 563 participants, a shared bias of that size is still
distinguishable from zero.

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

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">rep</th>
<th data-quarto-table-cell-role="th">b_f</th>
<th data-quarto-table-cell-role="th">P (b_f)</th>
<th data-quarto-table-cell-role="th">any impossible coef.
significant</th>
<th data-quarto-table-cell-role="th">n</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>1</td>
<td>0.037</td>
<td>0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>2</td>
<td>0.018</td>
<td>0.268</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>3</td>
<td>0.036</td>
<td>0.001</td>
<td>yes</td>
<td>562</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>4</td>
<td>0.019</td>
<td>0.166</td>
<td>no</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>5</td>
<td>0.039</td>
<td>0.001</td>
<td>yes</td>
<td>562</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5</td>
<td>6</td>
<td>0.036</td>
<td>0.003</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">6</td>
<td>7</td>
<td>0.044</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">7</td>
<td>8</td>
<td>0.031</td>
<td>0.045</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">8</td>
<td>9</td>
<td>0.030</td>
<td>0.029</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">9</td>
<td>10</td>
<td>0.037</td>
<td>0.003</td>
<td>yes</td>
<td>562</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">10</td>
<td>11</td>
<td>0.021</td>
<td>0.150</td>
<td>no</td>
<td>561</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">11</td>
<td>12</td>
<td>0.042</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">12</td>
<td>13</td>
<td>0.023</td>
<td>0.035</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">13</td>
<td>14</td>
<td>0.040</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">14</td>
<td>15</td>
<td>0.040</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">15</td>
<td>16</td>
<td>0.047</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">16</td>
<td>17</td>
<td>0.031</td>
<td>0.011</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">17</td>
<td>18</td>
<td>0.037</td>
<td>0.001</td>
<td>yes</td>
<td>562</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">18</td>
<td>19</td>
<td>0.045</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">19</td>
<td>20</td>
<td>0.028</td>
<td>0.021</td>
<td>yes</td>
<td>563</td>
</tr>
</tbody>
</table>

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

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">rep</th>
<th data-quarto-table-cell-role="th">b_f</th>
<th data-quarto-table-cell-role="th">P (b_f)</th>
<th data-quarto-table-cell-role="th">any impossible coef.
significant</th>
<th data-quarto-table-cell-role="th">n</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>1</td>
<td>0.041</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>331</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>2</td>
<td>0.051</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>323</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>3</td>
<td>0.032</td>
<td>0.010</td>
<td>yes</td>
<td>338</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>4</td>
<td>0.033</td>
<td>0.014</td>
<td>yes</td>
<td>340</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>5</td>
<td>0.055</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>317</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5</td>
<td>6</td>
<td>0.056</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>312</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">6</td>
<td>7</td>
<td>0.039</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>339</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">7</td>
<td>8</td>
<td>0.037</td>
<td>0.003</td>
<td>yes</td>
<td>346</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">8</td>
<td>9</td>
<td>0.031</td>
<td>0.016</td>
<td>yes</td>
<td>333</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">9</td>
<td>10</td>
<td>0.026</td>
<td>0.026</td>
<td>yes</td>
<td>323</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">10</td>
<td>11</td>
<td>0.054</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>339</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">11</td>
<td>12</td>
<td>0.057</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>328</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">12</td>
<td>13</td>
<td>0.049</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>335</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">13</td>
<td>14</td>
<td>0.038</td>
<td>0.004</td>
<td>yes</td>
<td>325</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">14</td>
<td>15</td>
<td>0.030</td>
<td>0.005</td>
<td>yes</td>
<td>333</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">15</td>
<td>16</td>
<td>0.051</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>322</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">16</td>
<td>17</td>
<td>0.039</td>
<td>0.002</td>
<td>yes</td>
<td>316</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">17</td>
<td>18</td>
<td>0.040</td>
<td>0.003</td>
<td>yes</td>
<td>332</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">18</td>
<td>19</td>
<td>0.044</td>
<td>&lt;0.001</td>
<td>yes</td>
<td>333</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">19</td>
<td>20</td>
<td>0.029</td>
<td>0.021</td>
<td>yes</td>
<td>325</td>
</tr>
</tbody>
</table>

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

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">Effect</th>
<th data-quarto-table-cell-role="th">Pooled estimate</th>
<th data-quarto-table-cell-role="th">Pooled P</th>
<th data-quarto-table-cell-role="th">Mean of participant-level
estimates</th>
<th data-quarto-table-cell-role="th">Two-step P</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>intercept</td>
<td>1.049</td>
<td>&lt;0.001</td>
<td>2.63</td>
<td>&lt;0.001</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>reward</td>
<td>0.203</td>
<td>&lt;0.001</td>
<td>0.58</td>
<td>&lt;0.001</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>previous transition</td>
<td>0.041</td>
<td>&lt;0.001</td>
<td>-0.32</td>
<td>0.102</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>reward:previous transition</td>
<td>0.159</td>
<td>&lt;0.001</td>
<td>0.44</td>
<td>&lt;0.001</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>future transition</td>
<td>0.002</td>
<td>0.843</td>
<td>-0.51</td>
<td>&lt;0.001</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5</td>
<td>reward:future transition</td>
<td>0.000</td>
<td>0.972</td>
<td>-0.07</td>
<td>0.620</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">6</td>
<td>previous transition:future transition</td>
<td>-0.012</td>
<td>0.130</td>
<td>0.19</td>
<td>0.330</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">7</td>
<td>reward:previous transition:future transition</td>
<td>0.007</td>
<td>0.382</td>
<td>-0.05</td>
<td>0.620</td>
</tr>
</tbody>
</table>

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

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">Analysis</th>
<th data-quarto-table-cell-role="th">Model fits</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>Real future transition, MLE</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>Simulated future transition (70/30), MLE</td>
<td>562</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>Simulated predictor, MLE, 20 replications</td>
<td>11256</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>Balanced predictor, MLE, 20 replications</td>
<td>11260</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>Intercept simulation (7 x 500 simulated partic...</td>
<td>3500</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5</td>
<td>Real future transition, Firth</td>
<td>563</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">6</td>
<td>Simulated predictor, Firth, 20 replications</td>
<td>11254</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">7</td>
<td>Simulated predictor, Firth, separated excluded...</td>
<td>6590</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">8</td>
<td>Pooled fit across all participants</td>
<td>1</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">9</td>
<td>Total</td>
<td>45549</td>
</tr>
</tbody>
</table>

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
