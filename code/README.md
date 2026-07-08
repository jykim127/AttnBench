# ACCS Benchmark — Core Code

Minimal pipeline to reproduce the ACCS faithfulness results. This is the core
subset (training → ACCS → statistics → readouts → chemical validation); the
figure-generation scripts are not included. Additional architecture variants
(D-MPNN, AttentiveFP) and auxiliary analyses (PR-AUC, mean-feature masking,
initialization variance) are available in the full repository.

## Setup

```
pip install -r requirements.txt
```

Datasets: obtain the seven MoleculeNet benchmarks (BBBP, BACE, HIV, ClinTox,
ESOL, FreeSolv, Lipophilicity) from MoleculeNet and place them under
`data/<dataset>/` as `train.csv`, `valid.csv`, `test.csv` with `smiles` and
label columns. Bemis–Murcko scaffold splits (seed 42 and seeds 0–4) follow the
sizes in `../scaffold_split_sizes.csv`.

## Pipeline (run order)

| Step | Script | Purpose |
|---|---|---|
| 1 | `02_train_models.py` | Train the ECFP4-MLP fingerprint model |
| 2 | `04_train_gnn.py` | Train the GCN, GIN, and GAT graph models |
| 3 | `03_accs_benchmark.py` | Compute ACCS with the true / permuted / random / degree-matched masking controls |
| 4 | `06_bootstrap_ci.py` | Bootstrap 95% CIs and Benjamini–Hochberg correction over the full test family |
| 5 | `08_xai_baselines_bootstrap.py` | Post-hoc readouts (Integrated Gradients, Grad×Input, GNNExplainer, SHAP) on the same trained weights |
| 6 | `12_chem_validation.py` | Chemical validation vs. Wildman–Crippen atomic logP and Brenk/PAINS structural alerts |

`train_gnn_utils.py` provides the shared atom featurization and model
definitions used by the graph scripts.

`ACCS_vs.perm` (the primary metric) = the drop in test performance (AUROC /
RMSE) from masking the top-`k`% saliency-ranked atoms, minus the drop from a
score-preserving permutation of the same atoms; positive values indicate a
faithful ranking. Default mask ratio = 0.2.

Precomputed numeric results for every table and figure are in the parent
folder (`../*.csv`, `../*.json`).
