# Relative Age Effect and Feedback Loops in Youth Talent Selection: a Simulation Study of Bias Amplification and Mitigation

**Student:** Gasbarro Samuele
**Email:** [name.surname@studio.unibo.it]
**Matriculation number (matricola):** [number]
**Course:** Ethics in AI, University of Bologna
**Document:** Project report (draft v0.2, work in progress)

---

## Abstract

Youth sport organisations select talent through multi-stage pathways (e.g., U12 to U18) and increasingly rely on data-driven tools. Annual age-grouping produces the Relative Age Effect (RAE): athletes born early in the selection year are systematically over-selected, because relative age is confounded with biological maturation. A model trained on past selection outcomes may inherit this bias and amplify it through a feedback loop: selected athletes receive more coaching and playing time, which improves their observed performance and reinforces later selections. Since real youth data are scarce and true talent is never observable, this project builds a seeded and tested Python simulator of a multi-stage selection pipeline in which latent talent is independent of birth date by construction. This makes it possible to measure directly the harm of biased selection, i.e., talented athletes born late in the year who are excluded. The study asks (RQ1) how much machine-learning selectors amplify the RAE compared with the bias already present in the data, (RQ2) whether the training-exposure feedback loop increases talent loss across stages and cohorts, and (RQ3) which mitigations (relative-age-adjusted scores, bio-banding, birth-quarter quotas, rotating selection slots) reduce talent loss and at what cost in selection efficiency. Results are reported with fairness metrics, efficiency metrics, multiple random seeds and confidence intervals. The project addresses the Trustworthy AI requirements of fairness, transparency and societal well-being, and uses no personal data. The simulator explores mechanisms and compares mitigations; it does not predict real-world outcomes.

---

## 1. Context and research questions

The Relative Age Effect (RAE) is a well-documented advantage for athletes born early in the
selection year (Cobley et al., 2009). A recent systematic review of machine learning in youth talent
identification recommends fairness audits across relative-age quartiles, sex and maturation
status (Tang et al., 2026). Real youth data are scarce and true talent is unobservable, so the harm
of biased selection cannot be measured directly; a simulation with known latent talent makes it
measurable.

- **RQ1.** How much does a selector trained on historical outcomes amplify the RAE, compared with
  the bias already present in the data?
- **RQ2.** Does the training-exposure feedback loop increase talent loss across successive stages
  and cohorts?
- **RQ3.** Which mitigations reduce talent loss, and at what cost in selection efficiency?

## 2. Model specification (draft)

**Athletes.** For athlete *i*, latent talent is T_i ~ N(0, 1), independent of birth date. The birth
date is U_i ~ Uniform[0, 1), the fraction of the selection year elapsed since the cut-off
(0 = oldest in the age group). The birth quarter Q_i is floor(4 U_i) + 1. Standardised relative age
is ra_i = (1/2 - U_i) / sqrt(1/12), and biological maturation is
M_i = rho * ra_i + sqrt(1 - rho^2) * e_i with e_i ~ N(0, 1), so M has unit variance and correlation
rho with relative age.

**Pipeline.** There are S stages. At stage s, all athletes still in the system (the pool) are
observed, and the top k_s = round(f_s * N) are selected, where f_s is a non-increasing fraction of
the initial cohort of N athletes. Observed performance is

    P_(i,s) = a * T_i + b_s * M_i + c * E_(i,s) + sigma * z_(i,s),   b_s = b_0 * delta^s,

where z is standard normal noise, delta < 1 makes the maturation effect fade at later stages, and
E_(i,s) = g * (number of earlier stages at which i was selected) is accumulated training exposure.
Selection at stage s therefore feeds exposure at stage s + 1, which closes the feedback loop.
Athletes not selected at a stage leave the system with probability p.

**Selectors.** (i) *Oracle*, ranking by T (benchmark, not implementable); (ii) *performance-based*,
ranking by P (human-like baseline). Planned: ML selectors trained on past selection decisions or on
later performance (RQ1, RQ2), and mitigations: relative-age-adjusted scores, bio-banding,
birth-quarter quotas, rotating selection slots (RQ3).

**Metrics (latent talent known).** With "high talent" meaning the top share of T:

- *Talent loss* by birth quarter: share of high-talent athletes not selected at the final stage;
  *equal-opportunity gap*: largest difference in talent loss across quarters.
- *Selection-rate ratio* Q1/Q4 among all athletes (1 means birth date plays no role).
- *Efficiency*: mean latent talent of the selected cohort; *regret*: gap to an oracle cohort of the
  same size.

**Invariants (tested).** The oracle has zero talent loss when the final selection share is at
least the high-talent share. With no maturation effect and no exposure effect, no RAE appears
(Q1/Q4 ratio close to 1). The ratio grows with the strength of maturation, and adding the exposure
effect increases it for the same athletes and noise.

## 3. Goals and deliverables

| Level | Goal | Deliverable |
|---|---|---|
| Core | Simulator and baseline selectors; answer RQ1 and RQ2 | Tested Python code, plots, confidence intervals over multiple seeds |
| Extended | At least two mitigations; sensitivity analysis (RQ3) | Pareto curves of talent loss vs efficiency; sensitivity study |
| Stretch | Explanation analysis; calibration to the literature | Feature-attribution study of birth-quarter proxies; calibrated RAE magnitude |

Trustworthy AI requirements addressed: fairness and non-discrimination (main focus), transparency
(open modelling assumptions), societal well-being (youth participation) and human agency
(mitigations support coaches and scouts rather than replace them). No personal data are used.

## 4. Implementation

Python package `rae` (NumPy, pandas, scikit-learn), tested with pytest and linted with ruff;
Makefile as build automation; CI on every push. Runs are seeded, and athletes, noise and dropout draws
come from separate streams so that selectors compared under the same seed face identical athletes
(common random numbers). See `README.md` for the repository layout.

## 5. Experiments and results

*To be completed.*

## 6. Open modelling decisions

- **Cut-off rule.** Which selection-year cut-off the Italian youth categories use, and how it
  affects the birth-quarter definition.
- **Parameter values.** Defaults in `SimConfig` are placeholders; each needs a justification or a
  sensitivity range (and, as a stretch goal, calibration to published RAE effect sizes).
- **Role of exposure.** With the current specification, exposure carries information about past
  selection in addition to creating bias, so it may reduce overall regret while increasing the Q1/Q4
  disparity. Decide whether exposure should be a pure bias, or also a real skill gain (in which case
  "latent talent" and talent loss need a more careful definition).
- **Labelling regimes** for the ML selectors: past selection decisions vs later performance.
- **Definition of "high talent"** (top 10% by default) and its sensitivity.

## 7. Limitations

The simulator is a stylised model: it explores mechanisms and compares mitigations, but its absolute
numbers depend on assumed parameters and do not predict real outcomes.

---

## References

- Cobley, S., Baker, J., Wattie, N., McKenna, J. (2009). Annual age-grouping and athlete development:
  a meta-analytical review of relative age effects in sport. *Sports Medicine*, 39(3), 235-256.
- Tang, Q., Wei, X., Tan, B. (2026). The Role of Machine Learning in Talent Identification for Team
  Sports: A Systematic Review. *Journal of Sports Science and Medicine*, 25, 58-83.
