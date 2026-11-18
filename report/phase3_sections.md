<!-- Draft text for report/report.md (phase 3). Merge it into the report by hand:
     sections 4.1-4.3 go under "4. Implementation", 5.1-5.2 under "5. Experiments and results",
     and the bullets at the end extend "6. Open modelling decisions".
     All numbers are PRELIMINARY: default placeholder parameters, linear models, 20 seeds
     (RQ1) and 8 seeds (RQ2). They must be re-run once parameters are justified. -->

### 4.1 Stage logs and learning selectors

Each run records, for every stage, the pool, the observed performance, the number of earlier
selections of each athlete and the selection decision. These logs are the "historical data"
on which machine-learning selectors are trained. Features are observed performance, number of
earlier selections (training exposure) and stage; birth information (position in the selection
year) can be withheld or added, to test fairness through unawareness. Two labelling regimes are
implemented: *decisions* (label = past selection decision) and *later performance* (label =
performance at the next stage, available only for athletes still in the system, a
selective-labels problem). Models are a linear/logistic model (default) and gradient boosting.

### 4.2 Retraining across cohorts

Generation 0 is historical data from a human-like (performance-based) selector. A model trained
on it is deployed on the next cohorts (generation 1), then retrained on its own decisions and
outcomes, and so on. Each cohort is simulated from an independent seed.

### 4.3 Additional invariants (tested)

A decision-imitation model without exposure reproduces the historical selection (overlap > 98%).
With no maturation and no exposure effect, ML selectors do not invent a RAE, in either regime,
with or without birth information, and the retraining loop shows no drift away from a ratio of 1.

### 5.1 RQ1: amplification by ML selectors (preliminary)

All selectors face the same test cohort (paired comparison). Q1/Q4 selection-rate ratio, mean over
20 seeds, 20,000 athletes per cohort:

| Selector | No feedback loop | With feedback loop |
|---|---|---|
| Oracle | 1.01 | 1.01 |
| Human-like (historical) | 2.74 | 3.44 |
| ML, decisions, no birth info | 2.74 | 3.41 |
| ML, decisions, birth info | 2.74 | 3.42 |
| ML, later performance, no birth info | 3.97 | 4.09 |
| ML, later performance, birth info | 6.31 | 6.13 |

### 5.2 RQ2: stages and cohorts (preliminary)

*Across stages* (human-like selector): the Q1/Q4 ratio falls from 3.07 to 2.76 over three stages
without the feedback loop (the maturation effect fades), and rises from 3.07 to 3.41 with it.
*Across cohorts*: moving from the human-like selector to an ML selector trained on later
performance causes a one-off level shift at generation 1; retraining on its own decisions for
four further generations does not compound the ratio further.

### Additional open modelling decisions

- **Decisions regime is trivially imitation.** The human-like selector is a deterministic function
  of the same features the ML model sees, so imitation is exact and cannot amplify anything. To make
  this regime informative, the historical selector needs scout noise or an explicit bias (e.g.,
  towards physically mature athletes).
- **Proxy through past selections.** The "later performance" model amplifies the RAE even without
  birth information, because the number of earlier selections is a proxy for early birth. This
  must be confirmed with a feature ablation.
- **No cross-cohort compounding.** Cohorts are independent, so the only memory across generations is
  the model. If compounding across cohorts is to be studied, a persistence mechanism is needed.
- **Selective labels.** Dropped-out athletes have no later performance; the effect of this on the
  trained models is not yet analysed.
