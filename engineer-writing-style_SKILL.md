---
name: engineer-writing-style
description: Use whenever drafting or editing the ISEE 2027 paper, its sections, abstracts, figure/table captions, or any technical write-up for the Pit 3 ground-vibration project. Enforces the clear, direct voice of a working mining and blasting engineer (a practitioner writing for practitioners) — plain, operational, and easy to read — while staying professional and publication-grade. Governs WRITING STYLE only; defer to the pit3-vibration-theory skill for all facts, numbers, and claims.
---

# Engineer Writing Style — Practitioner Voice (ISEE 2027)

This skill controls **how** the paper reads, not what it claims. The audience is the ISEE crowd: mine and blasting engineers, drill-and-blast supers, and technical services staff — people who run shots, not people who write journals. Write so a blasting engineer can read it once, understand it, and act on it.

> Facts, numbers, and findings always come from the **pit3-vibration-theory** skill. This skill only shapes tone and clarity.

## 1. Core voice
- Write like an experienced engineer explaining results to a competent colleague.
- Confident and direct. State what was found; don't tiptoe around it.
- Practical first: tie every result back to what it means in the pit (charge weight, distance, timing, compliance limit).
- Professional, but plain. No need to sound like a textbook to be credible.

## 2. Do
- Use **active voice**: "We removed the 10-row record" — not "The 10-row record was removed."
- Keep sentences **short** (aim ~15–20 words). One idea per sentence.
- Lead with the **point**, then the support. Don't bury the conclusion.
- Use **concrete, field terms**: charge per delay, burden, free face, scaled distance, PPV, compliance limit.
- Explain any statistic in **one plain clause** the first time it appears.
- Prefer numbers and units engineers use (kg, m, ms, mm/s), with imperial in parentheses for the ISEE audience.
- Use short paragraphs (2–4 sentences) and bullet lists where it aids scanning.

## 3. Don't
- No academic padding: avoid "it is important to note that," "the present study endeavors to," "a plethora of," "heretofore."
- No vague hedging stacks: avoid "may potentially possibly suggest." Say it once, plainly ("indicates", "shows", or "suggests").
- No jargon-for-jargon's-sake or unexplained acronyms.
- Don't nominalize: "we analyzed" not "an analysis was undertaken."
- Don't over-qualify findings into mush — state the finding, then its limit in a separate short sentence.
- Don't inflate. If an effect is small, say "small."

## 4. Vocabulary swaps (academic -> engineer)
| Academic | Engineer |
| :--- | :--- |
| utilize / employ | use |
| demonstrate / elucidate | show |
| in order to | to |
| exhibits a tendency to | tends to |
| is indicative of | shows / points to |
| a significant proportion of | most / about X% |
| subsequent to | after |
| prior to | before |
| facilitate | help / allow |
| ascertain | find / check |
| methodology (when you mean method) | method |
| in the vicinity of | near / about |

## 5. Handling technical & statistical terms
When a method or stat first appears, give a one-line plain meaning, then move on:
- "LOOCV (we test each blast with a model built from all the others)"
- "MAE of 0.031 g — on average the prediction is off by 0.031 g"
- "Sobol index — the share of the vibration scatter each factor explains"
Do not derail into derivations. If detail is needed, keep it brief and applied.

## 6. Practical framing (always close the loop)
Every result should answer: *so what, for the blast?* Examples:
- A sensitivity finding -> what a crew can adjust on the next shot.
- A prediction -> what charge weight or standoff it allows within the g <= 0.03 limit.
- A limitation -> what data is still needed.

## 7. Before -> After (project examples)
**Before (academic):**
> "A comprehensive sensitivity analysis was undertaken in order to ascertain the relative contributions of the operational parameters to the residual variance of the predicted vibration."
**After (engineer):**
> "We checked which blast-design factors actually drive vibration once distance and charge are accounted for."

**Before:**
> "The findings are indicative of the possibility that geological conditions may potentially exert a modest influence."
**After:**
> "Geology has a small but real effect: coal ground damps vibration."

**Before:**
> "The classical scaled-distance formulation demonstrates a statistically robust attenuation relationship with respect to charge weight and distance."
**After:**
> "The classic scaled-distance formula already captures most of the drop-off with distance and charge weight."

**Before:**
> "It is important to note that the row-number parameter does not exhibit a dominant contribution."
**After:**
> "Row number is not a major driver here."

## 8. Boundaries
- Stay **publication-grade**: clear and plain is not the same as casual or sloppy. No slang, no jokes, no first-name informality.
- Keep **ISEE format** requirements (English, dual metric/imperial units, section structure).
- **Never** trade accuracy for readability. If simplifying risks changing the meaning, keep the precise wording and add a plain-English gloss instead.
- **Defer to pit3-vibration-theory** for every factual claim, number, ranking, and reference. This skill never overrides that one.
