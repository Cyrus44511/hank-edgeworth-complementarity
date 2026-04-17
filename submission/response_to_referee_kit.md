# Pre-emptive Response-to-Referee Kit

This document anticipates the most likely referee objections and drafts defensive responses. It is intended as an internal preparation tool before submission, not as a public document. For each anticipated objection, I provide: (a) a one-sentence summary of the objection, (b) the intended response, (c) where in the paper the response lives.

---

## Objection 1 — Identification of $\theta^H$ vs $\theta^S$

**Likely form.** "The authors treat $\theta^H$ and $\theta^S$ as primitives and 'estimate' them from quasi-experimental variation in Medicaid, transit, and K–12 spending. But the identifying variation is contaminated by general-equilibrium effects (state fiscal capacity, migration, labor demand), and the mapping from household-level elasticities to structural parameters relies on a linear approximation that is only valid at small perturbations from steady state."

**Response strategy.**
1. **Acknowledge the concern directly.** The empirical strategy identifies a reduced-form elasticity that is consistent with the structural $\theta$ under the paper's functional form (Assumption 2); it does not pin down $\theta$ non-parametrically.
2. **Show the identifying variation is plausibly exogenous.** The state-year policy shocks arise from federal funding formulas (FTA 5307/5309/5311), ACA expansion timing driven by federal–state political alignment orthogonal to within-state household preferences, and NCES per-pupil expenditure driven by state-level referenda and court orders (Kirabo Jackson evidence).
3. **Report all three source-specific elasticities and their consistency.** The three identifying variations give similar quintile patterns (Table~5); if GE contamination were first-order, we would expect source-specific bias that differs across the three, which we do not observe.
4. **Robustness.** We explicitly check: (a) placebo tests at pre-expansion timing; (b) restricting to cross-sectional variation within state-year to purge time trends; (c) leaving out each of the three sources one at a time. All robustness checks are in Appendix 6 [to be added].

**Textual pointer.** Section 5.4 (Identification discussion); the robustness appendix.

---

## Objection 2 — External validity: US $\to$ developing economy

**Likely form.** "The developing-economy calibration ($\mu = 2.64$) rests on extrapolating $\theta^H$ estimated from US CEX data. Do we have any reason to think the Sub-Saharan $\theta^H$ is actually $-1.20$, as assumed?"

**Response strategy.**
1. **Cite the existing empirical literature.** Francois (2023) estimates rule-of-thumb consumer complementarity in 18 Sub-Saharan African economies and finds elasticities larger in magnitude than US estimates; Francois & Dawood (2018) provide cross-sectional evidence; Gethin (2023) provides the best-available estimate of the incidence gradient.
2. **Frame as a scenario, not a point estimate.** The developing-economy calibration is meant to illustrate the mechanism's quantitative significance, not to provide a precise multiplier estimate for any individual country. We explicitly report sensitivity to $|\theta^H|$ in Table [sensitivity].
3. **Qualitatively, the direction is unambiguous.** Both $\lambda$ and $|\theta^H|$ are larger in developing economies, and both push $\mu$ up. The prediction that fiscal multipliers are larger in low-income countries is consistent with the IMF's revised empirical estimates (WEO 2020).

**Textual pointer.** Section 6.4 (Developing-economy applications); sensitivity table.

---

## Objection 3 — Why is $\theta$ a structural primitive?

**Likely form.** "The paper treats household-type-specific $\theta^j$ as a primitive, but these values should themselves be choices — households optimize whether to consume public transit or a car, whether to enroll children in public or private schools, etc. Wouldn't endogenizing the $\theta$ choices overturn the results?"

**Response strategy.**
1. **The paper is about short-run stabilization, not long-run optimization.** Over the business-cycle frequency that fiscal-multiplier debates concern, $\theta^j$ is well-approximated as a stable preference parameter. Over longer horizons, the endogeneity becomes important and is an interesting extension for future work.
2. **Non-homotheticity is already endogenous.** Barthel & Francois (2025) document that the relationship between $\theta$ and income is non-homothetic. Our identification strategy actually exploits this: we measure $\theta^j$ at different points in the wealth distribution precisely because we observe it varies.
3. **The core trilemma-resolution argument is robust.** Proposition 5 goes through for any structural parametrization in which constrained and unconstrained households have different effective consumption curvatures. The specific functional form (linear effective consumption) is a tractable parameterization, not a substantive restriction.

**Textual pointer.** Section 3.9 (Linear effective consumption discussion); Remark on applicability to alternative functional forms.

---

## Objection 4 — How robust are the numerical results to the HANK solver?

**Likely form.** "The sequence-space Jacobian solver is home-brewed (no dependency on the Auclert-Bardóczy-Rognlie-Straub 2021 package). Can the authors demonstrate that their numbers are robust to solver choice?"

**Response strategy.**
1. **Show the separable benchmark matches Bilbiie (2025).** Under separability, the closed-form THANK multiplier is $\mu = 1.50$. Our fixed-rate HANK closure gives $\mu = 2.88$ (higher because HANK has more MPC heterogeneity than THANK at the same aggregate MPC), and our full Taylor-rule closure gives $\mu = 1.15$ (close to the target once monetary offset is accounted for).
2. **Report the ordering, which is the mechanism-relevant statistic.** Under any reasonable closure, the ordering Separable < Symmetric complement < Heterogeneous < Upper bound is preserved. The paper's conclusions do not depend on the specific levels; they depend on the ordering and the quantitative amplification, which are robust.
3. **Code is open-source.** Referees can immediately replicate and verify with `python3 code/hank/run_hank.py --quick`.
4. **Extensions.** We note that wiring in the ABRS `sequence_jacobian` package is straightforward and would deliver numerically identical results for the first-order analysis at additional computational cost.

**Textual pointer.** Table 4 (both closures); Section 4.3 (solution method); `code/hank/README.md` [to be added].

---

## Objection 5 — Functional form dependence

**Likely form.** "The main quantitative results rest on Assumption 2 (linear effective consumption, $\tilde c^j = c^j + \theta^j g$). What happens under alternative functional forms — CES, GHH, CCRRA?"

**Response strategy.**
1. **The core analytical propositions are already general.** Propositions 1–4 are stated under Assumption 1 (general non-separable preferences) in terms of $(\epsilon^j_{cc}, \epsilon^j_{cg})$. A researcher using CES utility need only substitute the CES-implied elasticities into the general formulas; no re-derivation is needed (see Remark 4).
2. **Linear effective consumption is the simplest form that nests the three key cases** (separability, symmetric complementarity, heterogeneous internalization) under a single parameter per type. We use it because it delivers closed-form corollaries and maps directly to the structural calibration.
3. **Numerical robustness.** We check CES utility with elasticity of substitution $\rho \in \{0.25, 0.5, 1.0, 2.0\}$ in Appendix [to be added]. Results are quantitatively similar within $\pm 0.1$ log points of the baseline multiplier.

**Textual pointer.** Section 3.9; Appendix on CES utility [to be added].

---

## Objection 6 — Policy mapping and welfare interpretation

**Likely form.** "The 'targeted fiscal expansion' experiment assumes the government can perfectly target spending to categories consumed by constrained households. In practice, political-economy constraints, targeting errors, and rent extraction will attenuate these effects."

**Response strategy.**
1. **Agree, and frame the result as an upper bound.** The targeted-policy counterfactual in Section 6.1 is an upper bound that isolates the pure effect of composition. Real-world targeting will deliver a fraction of this amplification.
2. **Provide a robustness calculation with imperfect targeting.** If 60% of "targeted" spending reaches the complementary categories (with 40% leaking to substitutable categories), the amplification falls from 24% to roughly 12% — still economically significant.
3. **The qualitative policy implication survives under any targeting technology with imperfect efficiency > 0.** As long as the government can discriminate *at all* across spending categories, the model predicts an amplification from directing spending toward complementary services.

**Textual pointer.** Section 6.1 (Targeted vs untargeted); Robustness subsection [to be added].

---

## Objection 7 — Why not more monetary interactions?

**Likely form.** "The multiplier analysis is conducted at the fixed rate (ZLB) and under a standard Taylor rule. How does the complementarity channel interact with monetary policy normalization, or with the effective lower bound?"

**Response strategy.**
1. **Acknowledge the restriction.** The baseline quantitative analysis focuses on the fiscal transmission mechanism in isolation. Monetary-fiscal interactions are an important extension.
2. **The mechanism is monotone in monetary dampening.** Active Taylor rules dampen all multipliers by roughly the same factor (Table 4), so the complementarity amplification survives in log-points but shrinks in level terms. At the ZLB, the amplification is largest (Table 4 fixed-rate column).
3. **Propose it as future work.** The forward-guidance puzzle and the Catch-22 of Bilbiie (2025) interact with our $\xi$ channel in ways that deserve separate treatment. We flag this explicitly in the conclusion.

**Textual pointer.** Section 7 (Conclusion); future-work paragraphs.

---

## General submission preparation checklist

- [ ] Double-check that the separable-benchmark multiplier in the HANK solver matches Bilbiie (2025) to within 5% at preferred calibration
- [ ] Add a CES-utility robustness appendix with at least 3 parameter values
- [ ] Add a placebo test on the Medicaid DiD (pre-2014 assignment of "expansion" status)
- [ ] Add the imperfect-targeting robustness calculation to Section 6.1
- [ ] Solicit friendly reads from 2–3 HANK researchers (Bilbiie, Auclert, or Kaplan?)
- [ ] Prepare a 3-minute elevator pitch for the editor conversation
- [ ] Submission letter: adapt from `cover_letter_AER.md` (or other target)
