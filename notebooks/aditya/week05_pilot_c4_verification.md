# Week 04–05 — Pilot construction + C4 acoustic-shortcut verification

**Author:** Aditya
**Date:** 2026-09-20
**Track:** S1 (benchmark construction + evaluation)
**Code:** `src/toolchain/construct_clip.py` (v1 orchestrator), `scripts/generate_benchmark.py`
(pilot generation), `scripts/audit_pilot_acoustic.py` (C4 verification)

---

## 1. What was built

- **Toolchain v1 orchestrator** (`construct_clip`): runs the v0 modules (splicer /
  crossfade / loudness / mixer) end-to-end on real sources. Samples 1–3 non-overlapping
  regions/component (0.4–2.0 s, 20 ms grid), builds each component track (bonafide base +
  spliced spoofed regions with loudness-matching C3 + raised-cosine crossfade), mixes at a
  sampled SNR (7–12 dB), and emits per-component 20 ms frame masks + a provenance record.
  Includes control **C1** (matched bonafide-into-bonafide twin: same seed → same region
  statistics, spliced with a different bonafide source → splice presence carries no label
  info). 7 unit tests pass.
- **Pilot generator** (`generate_benchmark.py`): pools real CompSpoofV2 sources by
  component × authenticity, generates clips per class (+ C1 twins), writes audio + masks +
  `manifest.csv`. Pilot: **1,600 mixed clips** (400/class) generated from `train` sources.

## 2. Purpose of this experiment (C4 verification)

The Week-2 follow-up audit found the **source** CompSpoofV2 clips leak the component-level
classes acoustically (speech 66%, env 72%, 4-class 49%), and flagged an open risk: if the
leak is spectral rather than level-based, loudness matching (C3) may not remove it.

This experiment runs the **same acoustic-only audit on the clips we constructed** to test
whether C3 removed the leak. (C1 twins excluded from the class-separation test — they are
bonafide by construction.)

## 3. Results — source vs constructed pilot

| Test | Chance | Source data | Constructed pilot | Δ (reduction) |
|:---|:---:|:---:|:---:|:---:|
| 4 mixed classes | 0.25 | 0.491 | **0.341** | −0.150 |
| Speech spoof-vs-bonafide | 0.50 | 0.661 | **0.559** | −0.102 |
| Environment spoof-vs-bonafide | 0.50 | 0.725 | **0.609** | −0.116 |

(Constructed-pilot classifiers: RF > LR in all three; n = 1,600; 5-fold CV.)

## 4. Interpretation

- **C3 works and matters.** Loudness/energy equalization during splicing reduced the
  acoustic shortcut substantially on every axis (10–15 points toward chance). The controls
  are doing real work — a naive splice would have inherited the full source leak.
- **Residual leak survives, worst on environment.** Speech (0.559) and 4-class (0.341) are
  near chance; **environment (0.609) is still meaningfully above chance (+0.109)**. This
  matches the Week-2 prediction: the residual is consistent with a **spectral** difference
  between generated and bonafide environmental audio that level-matching cannot remove.
- **Consistent with the project-wide pattern.** Environment is repeatedly the hardest/
  leakiest component (baseline env EER ≈ 0.43; source leak 0.72; residual pilot leak 0.61).
  Third independent piece of evidence for the same thread.

*Caveats:* the PASS/FAIL 0.10 threshold is arbitrary — the real signal is "large reduction,
small-to-moderate residual, worst for environment." Pilot uses free source pairing
(any spoof-env with any bonafide-speech); final pairing rules may shift these slightly.

## 5. Conclusion

Anti-shortcut construction reduces the source acoustic shortcut substantially but does **not
fully remove it for the environment component** (0.61 vs 0.50 chance). Loudness matching
alone is insufficient there; the residual is likely spectral.

## 6. Recommendation / next actions

1. **Add spectral matching for the environment component** (match the spliced segment's
   frequency envelope, not just loudness) — a "C3+" control — and re-run this audit.
2. If spectral matching cannot fully close it, **report the residual as a known benchmark
   limitation** and bound it via the C4 shortcut-probe ceiling in the paper.
3. **`audit_pilot_acoustic.py` is now the C4 verification tool:** re-run after any
   construction change; the environment number is the one to watch. Target: all three near
   chance.
4. Communicated to partner (owns the toolchain) so spectral matching can be added to C3.

## 7. Status

Toolchain v1 + pilot generation machinery complete and run on real audio; C4 verification
executed on a 1,600-clip pilot. Finding logged. Construction has a concrete improvement
target (spectral matching for environment).
