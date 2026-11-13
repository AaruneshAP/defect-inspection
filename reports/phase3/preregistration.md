# Phase 3c pre-registration

Written 2026-11-13 09:58:30 (UTC), before the test split was run through the model. Git HEAD at the time: `e583d2ff566a8d43d5300e193fcbf485df5bd4cd` (working tree has uncommitted Phase 3 code).

## Primary policy

- **Per-class review band** (lower, upper): `conf >= upper` alerts, `lower <= conf < upper` goes to a human, below `lower` is ignored.
- Tuned **once, on all 216 val images**, by `src.evaluate.tune_bands(..., budget=3)` (exact multiple-choice-knapsack DP; threshold grid `[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]` plus `off`).
- **Workload budget B <= 3**: alerts + reviews per image <= 3 on the val images.
- **reviewer_accuracy = 0.9** (symmetric; a missed-by-reviewer defect is a miss, a falsely confirmed detection is a false alarm).
- Cost model: expected cost as in `src.decision.policy_outcomes` (IoU 0.5 greedy class-aware matching, miss cost scaled by class weight x area factor).

### Frozen per-class bands (lower-upper)

| policy | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches |
|---|---|---|---|---|---|---|
| primary (a=0.9, B<=3) | 0.10-0.25 | 0.25-0.30 | 0.45 | 0.05-0.10 | 0.15-0.20 | 0.15 |
| secondary_a1.0 (a=1.0, B<=3) | 0.10-0.40 | 0.25-0.60 | 0.45-0.55 | 0.05-0.65 | 0.15-0.55 | 0.15-0.45 |
| secondary_a0.8 (a=0.8, B<=3) | 0.10 | 0.25-0.30 | 0.45 | 0.05 | 0.15 | 0.15 |

Exact values (JSON, also in `frozen.json`):

```json
{
  "primary": {
    "reviewer_accuracy": 0.9,
    "budget": 3,
    "bands": {
      "crazing": [
        0.1,
        0.25
      ],
      "inclusion": [
        0.25,
        0.3
      ],
      "patches": [
        0.45,
        0.45
      ],
      "pitted_surface": [
        0.05,
        0.1
      ],
      "rolled-in_scale": [
        0.15,
        0.2
      ],
      "scratches": [
        0.15,
        0.15
      ]
    }
  },
  "secondary_a1.0": {
    "reviewer_accuracy": 1.0,
    "budget": 3,
    "bands": {
      "crazing": [
        0.1,
        0.4
      ],
      "inclusion": [
        0.25,
        0.6
      ],
      "patches": [
        0.45,
        0.55
      ],
      "pitted_surface": [
        0.05,
        0.65
      ],
      "rolled-in_scale": [
        0.15,
        0.55
      ],
      "scratches": [
        0.15,
        0.45
      ]
    }
  },
  "secondary_a0.8": {
    "reviewer_accuracy": 0.8,
    "budget": 3,
    "bands": {
      "crazing": [
        0.1,
        0.1
      ],
      "inclusion": [
        0.25,
        0.3
      ],
      "patches": [
        0.45,
        0.45
      ],
      "pitted_surface": [
        0.05,
        0.05
      ],
      "rolled-in_scale": [
        0.15,
        0.15
      ],
      "scratches": [
        0.15,
        0.15
      ]
    }
  },
  "flat_matched": {
    "budget": 3,
    "threshold": 0.127
  },
  "flat_0.25": {
    "threshold": 0.25
  },
  "n_val_images": 216
}
```

## Secondary policies

Same procedure and budget, with reviewer_accuracy = 1.0 and 0.8. Reported alongside the primary; they do not replace it.

## Baselines on test

1. `flat 0.25`: one confidence threshold 0.25 for every class, alerts only.
2. **Matched flat**: the single flat threshold (alerts only) with the lowest val cost among those with alerts/image <= 3 on val; threshold = 0.127, searched on a 0.001 grid.
3. `no alerts`.

## Test protocol

- `python -m src.predict --split test --allow-test` is run once; `python -m src.phase3c test` applies the frozen thresholds above to the cached test detections. Nothing is retuned on test; the test command refuses to run twice.
- Metrics: cost (total and per image), recall (pooled expected caught / GT), alerts and reviews per image, with 95% percentile CIs from a 10,000-draw image bootstrap (seed 0); savings are paired bootstraps.
- Primary outcome: cost saving of the primary policy versus the matched flat baseline. It counts as supported only if its 95% CI lies entirely above 0.

## Cost configuration (illustrative assumptions)

```json
{
  "cost_miss": 10,
  "cost_false_alarm": 1,
  "cost_review": 0.3,
  "class_weights": {
    "crazing": 1.0,
    "inclusion": 1.0,
    "patches": 1.0,
    "pitted_surface": 1.0,
    "rolled-in_scale": 1.0,
    "scratches": 1.0
  },
  "area_severity": {
    "area_frac_small": 0.02,
    "area_frac_large": 0.2,
    "factor_small": 0.4,
    "factor_large": 2.0
  },
  "review_band_width": 0.15
}
```

## Provenance hashes (SHA-256)

- configs/cost.yaml: `04c76e1c531c0f71dc6fa5b77716689e2b77b1d6094f7d4eb0f2acc877f6001a`
- models/best.pt: `dff41d9a7f7be82791563f74990d04ffe12545767f6c8bd4c62d1100fcece56b`
- val detections cache: `2cea011e2c86e45a680a9cdff2c262164e8e32a696c3ed9dd32498d4639be7e9`
- src/decision.py: `2f46719b671a9443b9ab2fe6501eeb39b34206db8e56841c3abc6021ffde713b`
- src/evaluate.py: `387f891e40f0bd04a8ee577aa22922f9fec01e1e594960083245cd8eea5a2923`
