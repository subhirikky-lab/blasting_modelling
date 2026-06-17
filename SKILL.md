---
name: pit3-vibration-theory
description: Knowledge base and engineering guardrail for the Pit 3 Tanjung Enim ground-vibration prediction project (ISEE 2027). Use whenever writing or editing the paper, developing the web-app, or discussing the scaled-distance model, sensitivity analysis (Weight/OAT/Sobol), references, or any project theory — to keep all claims accurate, grounded in available sources, and within the agreed scope.
---

# Pit 3 Ground Vibration — Theory & Guardrail (ISEE 2027)

This document is the primary knowledge base and guardrail for AI-assisted technical writing and development for the ISEE 2027 ground-vibration prediction project at Tanjung Enim.

## 1. Project Overview
- **Title:** Ground Vibration Prediction at Tanjung Enim Mine Site: Extending Scaled Distance with Operational and Geological Parameters.
- **Site:** Pit 3, Tanjung Enim, South Sumatra, Indonesia.
- **Engineering limit:** peak acceleration g <= 0.03, where g = A_max / 9806.65.
- **Core objective:** extend the classical scaled-distance (SD) model with operational and geological parameters (via a weighting factor) to explain residual variance after distance and charge are accounted for.

## 2. Model Architecture (9-step pipeline — final, do not alter)
1. **Baseline:** SD = R / sqrt(W); A = k * SD^(-n). *(Source: Surface Blast Design — Konya 1990; Blasters Handbook)*
2. **Residual ratio:** actual / baseline.
3. **Weighting factor (WF_norm):** geometric mean of residual ratios per parameter level.
4. **Monte Carlo:** 10,000 iterations; sigma from Leave-One-Out Cross-Validation (LOOCV). *(Source: Monte Carlo Vibration Prediction Modelling, Basalt Hill Quarry NSW; v8n1_a02)*
5. **Mean estimate:** mean = median * exp(sigma^2 / 2) — corrects logarithmic retransformation bias.
6. **Design value:** median * lambda, where lambda is the empirical P90 from LOOCV. *(Source: Eurocode 7 — Geotechnical Design)*
7. **Sensitivity analysis:** Weight, One-At-a-Time (OAT), and first-order Sobol index Sᵢ = V[E[Y|xᵢ]] / V[Y]. *(Source: Benaïchouche & Rohmer 2016, building on Sobol 1993)*
8. **Diagnostics (NOT predictors):** dominant frequency and Scaled Depth of Burial (SDOB). *(Source: Surface Blast Design / Blasters Handbook — confirm which covers SDOB)*
9. **Error metrics:** MAE and Normalized MAE (NMAE). *(Source: Advantages of the MAE over the RMSE — Willmott & Matsuura)*

## 3. Theory & Justification
| Theory / Method | Description | Why used here | Grounding source (available) | Scope / Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **Scaled Distance** | Relates distance and charge weight to vibration attenuation. | Industry-standard baseline. | Surface Blast Design (Konya 1990); Blasters Handbook | Valid R >= 86 m |
| **Monte Carlo** | Iterative random sampling for probabilistic output. | Quantifies site-specific uncertainty. | Basalt Hill Monte Carlo report; v8n1_a02 | 10,000 iterations; N=26 |
| **Eurocode 7 design value** | Cautious (characteristic) value for safety. | Decisions "err on the safe side" at P90, not the mean. | Eurocode 7 — Geotechnical Design | Empirical lambda from LOOCV |
| **MAE over RMSE** | Mean absolute error. | Natural, unambiguous average error; robust to outliers. | Advantages of the MAE over the RMSE (Willmott & Matsuura) | Validation metric |
| **First-order Sobol** | Variance-based global sensitivity, Sᵢ = V[E[Y\|xᵢ]]/V[Y]. | Identifies which extended parameters drive residual variance. | Benaïchouche & Rohmer (2016); Sobol (1993) | First-order only |
| **Frequency analysis** | Spectral content of vibration. | Structural/slope resonance risk. | Surface Blast Design (confirm) | **Diagnostic only** |
| **SDOB** | Scaled depth of burial / confinement. | Confinement audit for design defensibility. | confirm available source | **Diagnostic only** |

## 4. Locked Decisions
- **Dataset:** 26 records / 19 blast events (cutoff: 14 June 2026).
- **Exclusion:** the 10-row record (21 March 2025) is excluded on **scope** grounds — Pit 3 has moved away from single 10-row rounds to staged rounds of <= 9 rows (broken muck acts as an additional free face and energy buffer). This is a scope decision, **not** a data error; the record is valid but out-of-regime.
- **Operational scope:** distance >= 86 m, g <= ~0.12, rows <= 9.
- **Pipeline:** Section 2 is final and must not be modified.

## 5. Key Findings (final, honest)
- **Convergence:** Weight, OAT, and first-order Sobol rankings are **convergent** — high confidence despite small N.
- **Top driver:** the **FREEFACE group** (Freeface Count + Freeface Echelon) most influences residual variance.
- **Control delay:** ranked #3, but its low sensitivity reflects near-constant values (~109 ms) in the dataset — a **data-variation limitation, not physical insignificance**.
- **Row number:** **NOT dominant** (~2.6%). Preliminary claims of ~66–70% were an **artifact** of a single excluded 10-row record and must not be repeated.
- **Geology:** modest overall (~0.9%), but **coal attenuation** (damping) is physically robust; normal (intact) rock transmits most, fault is near-neutral.
- **Abstract vs paper:** the submitted (locked) ISEE abstract states row number dominant (65.7%); this was **preliminary**. The full paper reports the finalized result and acknowledges the refinement (use "refine", not "error"). Do not reproduce the row-number-dominant claim.

## 6. Guardrails (ATURAN KERAS)
1. **Grounding:** explain theory only from the available source documents (Section 7A) or established foundational works (Section 7B). Do not invent new theories or external literature.
2. **No fabricated numbers:** never invent statistics or constants. If missing, state "perlu dikonfirmasi ke penulis".
3. **Honest claiming:** do not over-claim precision/rankings given N=26. Only assert findings stable across all three sensitivity methods.
4. **No preliminary claim:** never state row number is dominant.
5. **Geological caution:** faults can amplify or attenuate depending on geometry; only **coal damping** is treated as robust.
6. **Functional distinction:** always separate **Predictors** (SD + weighting factor) from **Diagnostics** (frequency, SDOB).
7. **Language/format:** paper in English, ISEE format.

## 7. Sources & Citation Policy

### 7A. Available source documents (grounding — what may be quoted/verified)
1. Abstract_Methodology sobol.pdf
2. Advantages of the mean absolute error (MAE) over... .pdf  (Willmott & Matsuura)
3. BLASTING_FOR_ROCK_EXCAVATIONS.pdf
4. Blasters handbook.pdf
5. Explosives_and_Blasting_Procedures_Manual.pdf
6. Forecasting blast-induced ground vibration developing a CART.pdf
7. MAE MAE.pdf
8. MONTE CARLO VIBRATION PREDICTION MODELLING AT PROPOSED BASALT HILL QUARRY NSW.pdf
9. Mine_Design_Planning_and_Sustainable_Exploitation.pdf
10. PERFORACION_Y_VOLADURA_DE_ROCAS_EN_MINERIA.pdf
11. Presplitting_and_Controlled_Blasting_Techniques.pdf
12. Surface_Blast_Design_by_Walter_and_Konya_1990.pdf
13. eurocode_7_-_geotechnical_design_EN_1997-1-2004.pdf
14. v8n1_a02.pdf
15. Benaïchouche & Rohmer (2016), "Sobol' indices and variance reduction diagram estimation..." (BRGM / HAL hal-01338344)

### 7B. Citation policy
- **Foundational methods** may be attributed to their originating works as standard practice (e.g., scaled distance — Duvall & Fogelson 1962 / Ambraseys & Hendron 1968; Sobol index — Sobol 1993), even when read via the textbooks in 7A.
- **Specific numbers and claims** must trace to a document in 7A or to the project data. Never cite a paper that has not been read for a specific quantitative claim.

*Role: act as a technical writing assistant and guardrail for the ISEE 2027 paper and web-app development.*

## 8. Glossary
- **Scaled Distance (SD):** distance normalized by sqrt of charge weight per delay.
- **Residual Ratio:** observed vibration / baseline-predicted vibration.
- **WF_norm:** geometric mean of residual ratios per parameter level, used to adjust the baseline.
- **LOOCV:** Leave-One-Out Cross-Validation; used to estimate sigma and out-of-sample accuracy on small datasets.
- **MAE / NMAE:** (normalized) mean absolute error; primary accuracy metric.
- **RMSE:** root-mean-square error; penalizes large errors, appropriate for Gaussian errors.
- **First-order Sobol index:** Sᵢ = V[E[Y|xᵢ]]/V[Y]; share of output variance attributable to a parameter.
- **OAT:** one-at-a-time local sensitivity (vary one input, hold others fixed).
- **Design Value:** conservative value = median * lambda (empirical P90 from LOOCV).
- **SDOB:** Scaled Depth of Burial; confinement diagnostic.
- **Retransformation Bias:** systematic error when converting a log-space fit back to linear units.
