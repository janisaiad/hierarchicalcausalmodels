# STAR — first HCM pass summary

## Chosen hierarchical schema

- unit `i`: `schoolidk`
- subunit `j`: student within school `i`
- latent unit variable: `U`
- subunit treatment `A_ij`: `1{stark == small}`
- subunit outcome `Y_ij`: `readk`
- observed unit regressors:
  - `S`: school type from `schoolk`
  - `M`: per-school mean `mathk`
  - `G`: per-school proportion female
  - `E`: per-school proportion white/asian
  - `L`: per-school proportion free lunch

This is a **first operational HCM approximation** of the variables used in `star_causal_discovery.ipynb`. The key modeling choice was to convert the non-treatment regressors into **unit-level school summaries** so that the HCM framework could be run with a subunit treatment and a subunit outcome.

## Data choices

- rows after dropping missing values on the selected STAR columns: `5768`
- schools used as units: `79`
- balanced students sampled per school: `34`

## First-pass results

| Graph | Status | Identifiable | ATE |
|---|---|---:|---:|
| `PC` | `ok` | yes | `0.0000` |
| `FCI` | `ok` | yes | `0.0000` |
| `ExactBIC` | `ok` | yes | `21357.6881` |
| `ConsensusMean` | `ok` | yes | `0.0000` |
| `DirectLiNGAM` | `too_slow_first_pass` | unknown | NA |

## Interpretation

These results should be read **very cautiously**.

### What this first pass does show

- the HCM pipeline can be connected to the `STAR` discovery graphs
- a concrete hierarchical schema can be instantiated with `schoolidk` as the unit
- identification succeeded on several translated graphs

### What is clearly unstable

- the estimated ATEs are **not stable across graphs**
- three graphs produce an ATE numerically equal to `0.0000`
- `ExactBIC` produces an extremely large ATE, which is not substantively credible for `readk`
- `DirectLiNGAM` is much slower than the others in this setup and was not completed in this first pass

### Likely reason

The main source of instability is the **choice of regressors and levels**:

- the discovery notebook is flat
- the HCM run requires a genuine hierarchical split between unit-level and subunit-level variables
- here, that split was imposed by turning `mathk`, `gender`, `ethnicity`, and `lunchk` into **school-level summaries**

This makes the analysis runnable, but it is still a **modeling compromise**, not a canonical HCM formulation of STAR.

## Main takeaway

The first pass is useful mainly as a **proof of connection** between:

- the graphs learned in `star_causal_discovery.ipynb`
- the HCM estimation framework in `examples/new/hcm_framework_test.ipynb`

My view on these results is the following.

They are useful as a **proof of concept**, but they are **not yet causally credible**. The important point is that we have successfully inserted `STAR` into the HCM pipeline, but the estimates are highly unstable. This suggests that the main problem is now clearly on the side of the **choice of hierarchy** and the **choice of regressors / levels**.

The most plausible diagnosis is:

- the hierarchy `school -> student` is too coarse to match the true treatment structure of `stark`
- converting `mathk`, `gender`, `ethnicity`, and `lunchk` into school-level aggregates is too strong a simplification

So the current results should not be read as reliable causal conclusions about `STAR`, but rather as evidence that the HCM machinery can be connected to this dataset and that the current modeling choices are not yet the right ones.

The next step should therefore be to redesign the HCM around a more faithful hierarchy, ideally:

- unit = class or teacher
- subunit = student

instead of using school-level aggregation as the main unit.
