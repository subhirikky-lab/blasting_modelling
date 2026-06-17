# ISEE 2027 — Defense Cheat Sheet & Reviewer Q&A
### Ground Vibration Prediction at Pit 3, Tanjung Enim
*Untuk Rikky — hafalkan intuisi + kalimat defense (English). Catatan dalam Bahasa Indonesia.*

---

## BAGIAN 1 — CHEAT SHEET CEPAT

| Konsep | Intuisi (1 kalimat) | Contoh data-mu | Kalimat defense (English) | Referensi |
|---|---|---|---|---|
| **LOOCV** | Uji model pakai data yang belum pernah "dilihat" → akurasi jujur | Buang 1 dari 26, latih di 25, prediksi yg dibuang, ulangi 26× | "Out-of-sample accuracy: each record is predicted by a model calibrated without it, avoiding in-sample optimism." | Stone (1974) |
| **MAE** | Rata-rata seberapa meleset (\|pred−aktual\|) | MAE 0.031 g; NMAE ~6% | "Mean absolute deviation between predicted and observed g — same unit as g, robust, interpretable." | Willmott & Matsuura (2005) |
| **RMSE** | Seperti MAE tapi hukum error besar lebih keras | cross-check | "RMSE penalizes large errors more; appropriate when errors are Gaussian." | Chai & Draxler (2014) |
| **Sobol** | % varians g yang dijelaskan tiap parameter (GLOBAL) | Freeface Count 36.5%, geologi 0.9% | "First-order Sobol indices decompose the variance of g, giving each parameter's standalone contribution across the full input space." | Sobol (1993); Saltelli (2008) |
| **OAT** | Ubah 1 parameter, lihat g berubah (LOKAL) | kolom OAT | "Local one-at-a-time perturbation; intuitive but ignores interactions — hence complemented by Sobol." | Saltelli (2008) |
| **Scaled Distance** | SD = R/√W, baseline klasik | Eq1–Eq2 | "Site-calibrated power law of charge weight and distance." | Duvall & Fogelson (1962); Ambraseys & Hendron (1968) |
| **Duan retransform** | Back-transform log → median; koreksi ke mean | mean = median × exp(σ²/2) | "Retransformation-bias correction when converting a log-space fit back to the expected value." | Duan (1983) |
| **Design Value** | Pakai percentile konservatif (P90), bukan mean | Design = median × λ | "Cautious characteristic-value philosophy for safety-critical decisions." | Eurocode 7 (2004) |

---

## BAGIAN 2 — SIMULASI PERTANYAAN REVIEWER + JAWABAN

### A. LOOCV
**Q1. Why LOOCV instead of a normal train/test split?**
> With only 26 records, a single split wastes data. LOOCV uses every record for both training and testing while keeping each prediction strictly out-of-sample, giving the most data-efficient unbiased error estimate.

**Q2. Isn't LOOCV unreliable for such a small dataset?**
> LOOCV is actually preferred for small datasets because it maximizes training data per fold. We report it as an honest out-of-sample indicator, not as a guarantee of performance on conditions outside the calibrated range.

**Q3. What does your LOOCV MAE of 0.031 actually mean?**
> On average, a prediction for a blast the model has never seen deviates from the measured g by about 0.031 — roughly 6% in normalized terms.

*(Catatan: kalau ditanya teori, cukup: "Cross-validation as formalized by Stone (1974)." Jangan masuk ke matematika.)*

---

### B. MAE / RMSE
**Q1. Why MAE and not RMSE as your primary metric?**
> MAE is in the same physical unit as g, is unambiguous, and is not inflated by a few large errors. We follow Willmott & Matsuura (2005). We also acknowledge Chai & Draxler (2014), who note RMSE is appropriate for Gaussian errors, so both perspectives are considered.

**Q2. Doesn't ignoring large errors hide bad predictions?**
> No — we additionally track underprediction and shortfall (the safety-relevant errors) separately, so large unsafe misses are not masked.

**Q3. What is NMAE?**
> Normalized MAE — MAE expressed relative to the data scale, allowing comparison independent of absolute magnitude.

---

### C. Sobol
**Q1. Why is Sobol your primary sensitivity measure?**
> It is global (covers the whole input space), accounts for parameter interactions, and is variance-based — valid for our non-linear power-law model. This follows Sobol (1993) and Saltelli et al. (2008).

**Q2. Is variance-based sensitivity meaningful with only 26 records?**
> We treat it as indicative, not definitive. That is exactly why we report three methods (weighted factor, OAT, Sobol) and only draw conclusions that are stable across all three.

**Q3. Freeface count is highest — is that robust?**
> Yes. Freeface count appears at the top in all three methods, and every level of that parameter is supported by at least two records, so it is not driven by a single observation.

**Q4. Earlier work suggested row number dominated — why not now?**
> *(JAWABAN PENTING — lihat Bagian 3, Q-Landmine #1)*

---

### D. OAT
**Q1. If OAT ignores interactions, why include it?**
> OAT gives an intuitive, operationally meaningful local sensitivity that field engineers relate to. It complements Sobol; agreement between the two strengthens confidence in the ranking.

**Q2. What reference supports your OAT?**
> One-at-a-time analysis is described as the basic local sensitivity method in Saltelli et al. (2008).

---

## BAGIAN 3 — PERTANYAAN "JEBAKAN" SPESIFIK PAPER-MU (paling penting!)

**Q-Landmine #1. Your abstract says row number dominates at 65.7%, but the paper says it does not. Explain.**
> The dataset and calibration were finalized after abstract submission. The finalized analysis showed the earlier row-number dominance originated from a single out-of-scope record — a 10-row blast, a configuration no longer used at Pit 3, which now stages large rounds into smaller ones. Once restricted to the current operational regime (up to nine rows), operational timing parameters emerge as most influential. The abstract reflects the preliminary analysis; the paper reports the finalized result.

**Q-Landmine #2. Did you remove the outlier just to improve results?**
> No. The record was excluded on operational grounds — it represents a discontinued 10-row blasting practice outside the model's intended scope. The decision is documented, and we report results with and without it for transparency.

**Q-Landmine #3. 26 records is very small. How reliable is this?**
> We make no claim of universal generality. The model is calibrated and validated for Pit 3 within a stated range (distance ≥ 86 m, g ≤ ~0.12, up to nine rows). LOOCV provides an honest in-range accuracy estimate, and the web tool is continuously updated as new blasts are recorded.

**Q-Landmine #4. Geology contributes <1% — does geology not matter?**
> Geology shows a modest but physically consistent effect: intact (normal) rock transmits vibration most efficiently, coal seams attenuate it, faults are near-neutral. The low sensitivity index reflects limited variability in the recorded geological classes, not a claim that geology is physically irrelevant.

**Q-Landmine #5. Control delay is 109 ms in most records — is it a real variable?**
> Control delay has effectively standardized to 109 ms in current Pit 3 practice, so it appears near-constant in the data and cannot be resolved by the sensitivity analysis. This reflects operational standardization rather than physical insignificance.

**Q-Landmine #6. How is your "extended" model better than classical scaled distance?**
> The classical two-parameter model leaves systematic residual scatter. By adding operational and geological parameters through a normalized weighting factor, we explain part of that residual and improve prediction accuracy, giving engineers a calculated compliance margin instead of a fixed conservative charge limit.

---

## BAGIAN 4 — CHECKLIST BACA JUJUR (sebelum ISEE)

- [ ] **Willmott & Matsuura (2005)** — baca penuh (4 hlm, open access)
- [ ] **Chai & Draxler (2014)** — baca penuh (4 hlm, open access)
- [ ] **Sobol (1993)** — paham konsep via cheat sheet + Wikipedia "Variance-based sensitivity analysis"
- [ ] **Stone (1974)** — paham konsep cross-validation (tak perlu baca penuh)
- [ ] **Saltelli (2008)** — baca bab intro (OAT vs global SA)
- [ ] **Siskind (1980)** & **Duvall & Fogelson (1962)** — skim bagian scaled distance & kriteria (public domain)
- [ ] **Floyd / Lucca / Tobin / Ambraseys & Hendron** — domain blasting, skim (sudah familiar)

> **Prinsip:** Hanya cantumkan referensi yang sudah kamu skim cukup untuk tahu isinya & kenapa dipakai. Paham intuisi + 1 kalimat defense = cukup untuk metode statistik; untuk domain blasting kamu sudah ahli.
