"""
Verify paper alg:marginalize eligibility vs gallery hand-picked Q^C.

Builds each case: collapse -> augment (no marginalize, no X->Y gallery patch).
Computes marginalization_Qc_eligible_parents(aug, q_hat) and checks that every
gallery ``marginalize`` set is a subset of the eligible set (Q-restricted
sole-child rule; see do_calculus.marginalization_Qc_eligible_parents).

Run from repo root:
  uv run python examples/new/verify_marg_conditions.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_EX = _ROOT / "examples" / "new"
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from collapsed_cases import COLLAPSED_DO_CALCULUS_CASES, _make_hscm  # noqa: E402
from hierarchicalcausalmodels.do_calculus import (  # noqa: E402
    augment_collapsed_model,
    collapse,
    marginalization_Qc_eligible_parents,
    marginalize_augmented_model,
)


def augmented_cgm_before_marginalize(case):
    """Collapsed CGM after augment only (no marginalize, no gallery X->Y patch)."""
    (_name, nodes, edges, unit_nodes, subunit_nodes, augment, _marginalize, _Y, _X, _u, _exp) = case
    hscm = _make_hscm(nodes, edges, unit_nodes, subunit_nodes)
    cgm = collapse(hscm)
    if augment is not None:
        q_hat, parents = augment
        cgm = augment_collapsed_model(cgm, q_hat, parents)
    return cgm


def main() -> int:
    failures: list[str] = []

    print("Case | q_hat | eligible Q^C (auto) | manual Q^C (gallery) | check")
    print("-" * 100)

    for case in COLLAPSED_DO_CALCULUS_CASES:
        cname = case[0]
        marginalize = case[6]
        augment = case[5]

        if augment is None:
            print("{:22} | {:5} | {:30} | {:20} | {}".format(cname, "—", "—", "—", "no augment"))
            continue

        q_aug, _ = augment
        cgm_aug = augmented_cgm_before_marginalize(case)
        eligible = marginalization_Qc_eligible_parents(cgm_aug, q_aug)

        if marginalize is None:
            print(
                "{:22} | {:5} | {:30} | {:20} | {}".format(
                    cname,
                    q_aug,
                    str(sorted(eligible)) if eligible else "[]",
                    "—",
                    "no marginalize",
                )
            )
            continue

        q_mar, manual_qc = marginalize
        if q_mar != q_aug:
            msg = "{}: marginalize q_hat {!r} != augment q_hat {!r}".format(cname, q_mar, q_aug)
            failures.append(msg)
            print("{:22} | {:5} | {:30} | {:20} | FAIL".format(cname, q_mar, str(sorted(eligible)), str(sorted(manual_qc))))
            continue

        bad = manual_qc - eligible
        strict = manual_qc == eligible
        if bad:
            failures.append(
                "{}: manual has ineligible {} (eligible={})".format(
                    cname,
                    sorted(bad),
                    sorted(eligible),
                )
            )
            tag = "FAIL"
        else:
            tag = "OK (manual == eligible)" if strict else "OK (manual ⊂ eligible)"
        print(
            "{:22} | {:5} | {:30} | {:20} | {}".format(
                cname,
                q_mar,
                str(sorted(eligible)),
                str(sorted(manual_qc)),
                tag,
            )
        )

    print("\n--- Optional: DAG compare marginalize(manual) vs marginalize(eligible) ---")
    for case in COLLAPSED_DO_CALCULUS_CASES:
        marginalize = case[6]
        augment = case[5]
        if marginalize is None or augment is None:
            continue
        cname = case[0]
        cgm_aug = augmented_cgm_before_marginalize(case)
        q_mar, manual_qc = marginalize
        eligible_set = marginalization_Qc_eligible_parents(cgm_aug, q_mar)
        if not eligible_set:
            failures.append("{}: empty eligible set with non-None marginalize".format(cname))
            continue
        g_man = marginalize_augmented_model(copy.deepcopy(cgm_aug), q_mar, manual_qc)
        g_full = marginalize_augmented_model(copy.deepcopy(cgm_aug), q_mar, set(eligible_set))
        same = set(g_man.dag.nodes) == set(g_full.dag.nodes) and set(g_man.dag.edges) == set(
            g_full.dag.edges
        )
        if same:
            print("{}: full eligible marginalization equals gallery manual".format(cname))
        else:
            print("{}: strict subset: gallery removes fewer Q nodes than full eligible set".format(cname))

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" ", f)
        return 1

    print("\nAll checks passed: manual Q^C ⊆ eligible for every case with marginalize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
