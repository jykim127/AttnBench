# AttnBench
# ACCS Benchmark — Data Release

Numeric data underlying every table and figure in *Readout-Dependent
Faithfulness in Molecular Graph Neural Network Explanations*. Values are the
final, Benjamini–Hochberg-corrected results reported in the manuscript and its
Supporting Information. Figure-generation code is not included.

`ACCS_vs.perm` is the primary faithfulness metric: the drop in test-set
performance (AUROC for classification, RMSE for regression) when the top-`k`%
saliency-ranked atoms are masked, minus the drop under a score-preserving
permutation of the same atoms. Positive values indicate a faithful ranking.
Unless noted, mask ratio = 0.2 and the six scaffold splits are seed 42 and
seeds 0–4.

## Files

| File | Content | Manuscript location |
|---|---|---|
| `accs_vs_perm_pooled.csv` | Native-readout ACCS_vs.perm pooled over 6 splits (mean, SD, # BH-significant splits/6) per model × dataset | Fig. 3, Table 3 |
| `predictive_performance.csv` | Test AUROC (classification) / RMSE (regression) per model × dataset | Table 2 |
| `per_split_accs.csv` | ACCS_vs.perm for every model × dataset × scaffold-split, with raw and BH-adjusted p-values | Table S4 |
| `scaffold_split_sizes.csv` | Test-set size per dataset for each of the six scaffold-split seeds | Table S2 |
| `gat_posthoc_multiseed.csv` | GAT post-hoc readouts (Integrated Gradients, Grad×Input) pooled over 6 splits | Fig. 2, Table S7 |
| `attentivefp_multiseed.csv` | AttentiveFP gradient and native-attention readouts pooled over 6 splits | Table 5, Table S8 |
| `dmpnn_multiseed.csv` | D-MPNN native (gradient) readout pooled over 6 splits | Table 3 |
| `mask_ratio_sweep.csv` | # BH-significant dataset–split combinations (of 42) per model at mask ratios 0.1 / 0.2 / 0.3 | Table S6 |
| `prauc_accs.csv` | # BH-significant splits (of 6) per model × classification dataset using PR-AUC instead of AUROC | Table S9 |
| `chem_validation_summary.csv` | Per-molecule Spearman ρ (attribution vs. Wildman–Crippen atomic logP) and structural-alert enrichment | Fig. 4A/B, chem-validation table |
| `chem_validation_dotplot_data.csv` | Tidy Spearman / enrichment values with per-molecule counts and s.e.m. | Fig. 4A/B |
| `example_molecule_attributions.json` | Per-atom ground-truth (Crippen), GCN-gradient, ECFP4-gradient, and GAT-attention scores for the representative BACE inhibitor | Fig. 4C |

## Column notes

- `n_sig_bh_of6` / `n_sig_bh6`: number of the six scaffold splits in which
  ACCS_vs.perm is Benjamini–Hochberg-significant (`q < 0.05`).
- `significant` in `per_split_accs.csv`: `*`, `**`, `***`, or `ns` (raw one-sided
  bootstrap p-value; `p_bh` is the BH-adjusted value over the 210-test family).
- Datasets are the seven MoleculeNet benchmarks; obtain the molecules from
  MoleculeNet and apply the splits in `scaffold_split_sizes.csv` (full split
  assignments are in the code repository).

## Citation

If you use these data, please cite the manuscript and MoleculeNet (Wu et al.,
*Chem. Sci.* 2018).
