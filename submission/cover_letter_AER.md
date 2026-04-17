# Cover Letter — American Economic Review

---

Bright Quaye
Department of Economics
Washington University in St. Louis
b.quaye@wustl.edu

[Date]

Professor [Editor Name]
Editor, *American Economic Review*
American Economic Association

---

Dear Professor [Editor Name]:

I am pleased to submit for consideration in the *American Economic Review* my manuscript, *On Heterogeneity and Edgeworth Complementarity Between Public and Private Consumption: Implications for the Fiscal Multiplier*.

The paper asks — and answers — a question that has been open since the original Bouakez–Rebei (2007) debate on government-spending multipliers: how does the distributional incidence of public services shape aggregate fiscal transmission? The paper's answer is that it matters a great deal, and that the answer requires a mechanism which has been absent from the modern HANK literature.

## Contribution

The paper embeds Edgeworth complementarity between public and private consumption into the tractable heterogeneous-agent New Keynesian framework of Bilbiie (2025, *ReStud*). The extension is parsimonious — a single cross-partial $U_{cg}^j \neq 0$ in household preferences — but the consequences are sharp:

1. **A new sufficient statistic.** Closed-form propositions deliver two sufficient statistics for fiscal transmission ($\chi$, $\xi$) that can be independently identified from household consumption responses and reduced-form VAR evidence. This is a direct generalization of Bilbiie's univariate $\chi$ statistic.

2. **A new resolution of the Auclert–Bardóczy–Rognlie (2024) trilemma.** When constrained and unconstrained households internalize public goods differently ($\theta^H \neq \theta^S$), the aggregate MPC decouples from the MPE through $\theta$-induced curvature heterogeneity, independently of the hand-to-mouth share. This is the first trilemma resolution that operates through the government-spending margin rather than labor-supply frictions, and is orthogonal to the Auclert et al. (2024) sticky-wage resolution and the Bilbiie–Hanks–Lavender (2025) $c$–$n$ complementarity resolution.

3. **A distributional dominance condition.** Fiscal policy is most effective precisely in economies where inequality is highest and where the poor most strongly complement public services with private consumption. This mechanism is not additive but multiplicative in inequality and complementarity, aligning the equity and efficiency of fiscal expansion.

## Quantitative implications

Taking the model to a full HANK calibration disciplined by US micro data (CEX 1996–2019 matched with state-year variation in Medicaid expansion, FTA transit funding, and NCES K–12 expenditure), we find:

- Plausibly heterogeneous internalization raises the impact multiplier to **1.70–1.84**, relative to **1.15** in the separable benchmark — a 48–62% amplification that accounts for roughly a third of the dispersion in the VAR-based empirical literature (Ramey 2019).
- Targeting fiscal expansion at complementary public services (health, transit, education) raises the multiplier another **24%** over untargeted spending, generating a consumption-equivalent welfare gain for the bottom quintile of **1.8%**.
- Austerity through cuts to complementary public services is **44% more contractionary** than the separable benchmark predicts, with welfare costs falling sharply on the bottom of the wealth distribution.
- In developing economies with higher $\lambda$ and stronger $|\theta^H|$ (Gethin's "triple curse" configuration), the impact multiplier reaches **2.64**.

## Why AER

The paper fits *AER* for four reasons. First, it addresses a question — the fiscal multiplier — that is central to macroeconomic policy and is of intrinsic interest to the broad readership. Second, it resolves a recent puzzle (the MPC/MPE trilemma) that was published in *AER: Insights* (Auclert, Bardóczy, and Rognlie 2024) and has generated significant follow-up work. Third, the empirical identification strategy combines CEX household data with quasi-experimental variation in three distinct policy instruments, providing the kind of empirical discipline that *AER* has increasingly required of quantitative macroeconomic papers. Fourth, the developing-economy implications speak directly to a policy debate that has received significant recent attention in both academic and IMF/World Bank circles.

## Data, code, and replication

A complete replication package accompanies the submission at `https://github.com/Cyrus44511/hank-edgeworth-complementarity`. This includes:
- Full LaTeX source for the paper and its appendices
- A sequence-space Jacobian HANK solver (pure NumPy, no external dependencies)
- The empirical identification pipeline in Python, run end-to-end in under 20 seconds
- Detailed download instructions for the four public data sources (BLS CEX, KFF Medicaid, FTA transit, NCES K–12)

The paper has not been submitted elsewhere. It has not been previously circulated in its current form, though earlier versions of the analytical results were presented at [list any seminars].

Thank you for considering the manuscript. I would be grateful for the editorial board's evaluation and look forward to engaging with any suggestions it might produce.

Sincerely,

Bright Quaye
Washington University in St. Louis
