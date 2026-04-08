# STAR — first hierarchical HCM pass

## Chosen schema

- unit `i`: `schoolidk`
- subunit `j`: student within school `i`
- latent unit confounder: `U`
- treatment `A_ij`: `1{stark == small}`
- outcome `Y_ij`: `readk`
- observed unit covariates: `S` = `schoolk`, `M` = school mean `mathk`, `G` = school proportion female, `E` = school proportion white/asian, `L` = school proportion free lunch

This first pass keeps the discovery graphs but moves the non-treatment regressors to the unit level as school summaries so that the HCM pipeline remains estimable with a subunit treatment and a subunit outcome.

## Data choices

- rows after dropping missing values: `5768`
- schools used as units: `79`
- balanced students per school: `34`

## Graph-by-graph HCM results

| Graph | Status | Identifiable | ATE | E[do(1)] | E[do(0)] |
|---|---|---:|---:|---:|---:|
| `PC` | `ok` | yes | 0.0000 | 436.1474 | 436.1474 |
| `FCI` | `ok` | yes | 0.0000 | 436.1474 | 436.1474 |
| `DirectLiNGAM` | `ok` | yes | -1884.8232 | 41094.6596 | 42979.4828 |
| `ExactBIC` | `ok` | yes | 20684.1248 | 426106.0222 | 405421.8973 |
| `ConsensusMean` | `ok` | yes | 0.0000 | 436.1474 | 436.1474 |

## Notes

### PC

- note: PC avec orientation heuristique des arêtes non orientées selon un ordre causal fixe.
- status: `ok`
- formula: `$P\left(Qy\right)$
- explanation: Do-calculus identification succeeded.

### FCI

- note: FCI ne fournit ici qu’un squelette; orientation heuristique imposée pour obtenir un DAG HCM exécutable.
- status: `ok`
- formula: `$P\left(Qy\right)$
- explanation: Do-calculus identification succeeded.

### DirectLiNGAM

- note: Arcs dirigés repris directement de DirectLiNGAM.
- status: `ok`
- formula: `$\sum_{E,G,L,M,S}{P\left(S\mid E\right) \cdot \left(\sum_{Qa}{P\left(Qa\right) \cdot P\left(Qy\mid G,L,M,Qa\right)}\right) \cdot P\left(M\mid E,G,L,Qa\right) \cdot P\left(L\mid E,S\right) \cdot P\left(G\right) \cdot P\left(E\right)}$
- explanation: Do-calculus identification succeeded.

### ExactBIC

- note: DAG score-based repris directement de ExactBIC.
- status: `ok`
- formula: `$\sum_{E,G,Qy_a,S}{P\left(S\mid E\right) \cdot P\left(Qy_a\mid G,S\right) \cdot P\left(Qy\mid Qa,Qy_a\right) \cdot P\left(G\right) \cdot P\left(E\right)}$
- explanation: Do-calculus identification succeeded.

### ConsensusMean

- note: Graphe moyen majoritaire d’après `preliminar_results.md`.
- status: `ok`
- formula: `$P\left(Qy\right)$
- explanation: Do-calculus identification succeeded.
