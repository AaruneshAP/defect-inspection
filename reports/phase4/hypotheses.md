# Phase 4a hypotheses (written before running src/analysis.py)

Model: models/best.pt (e0 baseline). Policy: single global confidence threshold 0.127 (reports/phase3/frozen.json).
Data: val split only (216 images, cached detections at conf floor 0.001). Nothing is retuned.

## H1: crazing is hurt by box ambiguity

Crazing AP50 rises substantially at IoU 0.3 relative to other classes.

Define `gain_c = AP30_c - AP50_c` for each class c. H1 is **supported** only if all hold:
1. crazing has the largest gain of the 6 classes;
2. crazing gain >= 0.10 absolute;
3. crazing gain >= 2x the median gain of the other 5 classes.

Otherwise H1 is not supported. (If crazing has the largest gain but misses 2 or 3, report it as "weakly / partially supported".)

## H2: crazing misses are localization failures, not absent detections

Among crazing GT boxes that are not true positives at threshold 0.127, most are "poorly localized"
(best same-class IoU with a kept detection in (0.1, 0.5)) rather than "missed" (no kept detection with IoU > 0.1).

H2 is **supported** if poorly-localized crazing GT boxes are > 50% of all non-TP crazing GT boxes
(and outnumber the "missed" ones). Otherwise not supported.

## Caveats stated up front

- Val has few crazing boxes, so shares and AP gains are noisy. Numbers are descriptive, with no significance test, and are reported as such.
- Categories use the kept detections only (conf >= 0.127), AP uses all cached detections (conf >= 0.001).

---

# Results (appended after running `python -m src.analysis`; the text above is unchanged)

**H1: supported.** AP30 - AP50 (val, all cached detections): crazing 0.688 - 0.425 = **+0.263**; inclusion +0.112,
patches +0.069, pitted_surface +0.045, rolled-in_scale +0.196, scratches +0.011 (median of the other five = 0.069).
Crazing is the largest gain, >= 0.10, and 3.8x the median of the others. Caveat: rolled-in_scale (+0.196) is not far behind (ratio 1.3).

**H2: supported.** Crazing GT (n=84) at conf >= 0.127: 43 TP, 37 poorly localized, 4 missed, 0 wrong class, 0 shared.
Poorly localized = 37 of 41 non-TP = **90%**; missed = 4 (10%). The 37 have best same-class IoU 0.15-0.50 (median 0.34; 27 of 37 are >= 0.3, 12 are >= 0.4).
