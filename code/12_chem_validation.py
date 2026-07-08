"""
12_chem_validation.py
=====================
Chemical validation of atom-level attributions against domain knowledge.

Two analyses (choose with --analysis):

  crippen : For regression datasets (Lipophilicity primary; ESOL/FreeSolv
            secondary), correlate each atom's attribution with the magnitude of
            its RDKit Crippen atomic logP contribution (Wildman--Crippen). If a
            method identifies chemically meaningful atoms, high-|contribution|
            atoms should receive high attribution. Reports the per-molecule
            Spearman correlation, averaged over the test set.

  alerts  : For classification datasets (BACE, ClinTox, HIV), test whether
            top-attribution atoms are enriched in structural-alert substructures
            (RDKit Brenk + PAINS FilterCatalogs). Reports precision@k and an
            enrichment ratio (precision@k / overall alert-atom fraction),
            averaged over molecules that contain at least one alert.

실행 (seed 42 모델 사용):
  cd /path/to/AttnBench
  # Crippen (회귀)
  for m in ecfp4 gcn gin gat; do
    python experiments/12_chem_validation.py --analysis crippen --dataset Lipophilicity --model $m
  done
  # Structural-alert enrichment (분류)
  for ds in BACE ClinTox HIV; do
    for m in ecfp4 gcn gin gat; do
      python experiments/12_chem_validation.py --analysis alerts --dataset $ds --model $m
    done
  done

출력:
  results/chem_validation/crippen_<dataset>_<model>.json
  results/chem_validation/alerts_<dataset>_<model>.json
  results/chem_validation_summary.csv   (누적)
"""

import argparse
import importlib.util
import json
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from scipy.stats import spearmanr

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("bootstrap_ci_06", _HERE / "06_bootstrap_ci.py")
bci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bci)
sys.path.insert(0, str(_HERE))

MASK_TOPK = 0.2   # top-k fraction for precision@k (matches the ACCS mask ratio)


# ══════════════════════════════════════════════════════════════════════════════
# Attribution (native saliency, seed-42 model) — reuse existing formulas
# ══════════════════════════════════════════════════════════════════════════════
def load_model_and_data(dataset, model_type, data_dir, result_dir, device):
    """Return (model, list[(mol, graph_or_None, smiles)]) for the seed-42 model."""
    from train_gnn_utils import DATASETS, smiles_to_data, GCNModel, GATModel, GINModel
    task = DATASETS[dataset]["task"]
    data_path = Path(data_dir) / dataset
    test_df = pd.read_csv(data_path / "test.csv")

    if model_type == "ecfp4":
        model_dir = Path(result_dir) / dataset / "mlp"
        model = bci.MLP()
        # 02_train_models.py 는 {"state_dict":...} 로 감싸 저장 (06과 동일 언랩)
        ckpt = torch.load(model_dir / "best_model.pt", map_location=device)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        model.to(device).eval()
        items = []
        for _, row in test_df.iterrows():
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is not None:
                items.append((mol, None, str(row["smiles"])))
        return model, items, task

    in_dim = None
    model_dir = Path(result_dir) / dataset / f"gnn_{model_type}"
    items, graphs = [], []
    for _, row in test_df.iterrows():
        g = smiles_to_data(row["smiles"], row[DATASETS[dataset]["label_col"]], task)
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if g is not None and mol is not None and mol.GetNumAtoms() == g.x.shape[0]:
            items.append((mol, g, str(row["smiles"])))
            if in_dim is None:
                in_dim = g.x.shape[1]
    if model_type == "gcn":
        model = GCNModel(in_dim)
    elif model_type == "gin":
        model = GINModel(in_dim)
    else:
        model = GATModel(in_dim, hidden_dim=64, heads=4)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.to(device).eval()
    return model, items, task


def atom_attribution(model, model_type, mol, g, smiles, device):
    """Per-atom attribution score (higher = more important)."""
    if model_type == "ecfp4":
        fp, bi = bci.smiles_to_ecfp4(smiles)
        fp_t = torch.tensor(fp, dtype=torch.float32, device=device)
        return bci.ecfp4_atom_saliency(model, fp_t, bi, mol.GetNumAtoms(), device)
    x = g.x.to(device)
    edge_index = g.edge_index.to(device)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    if model_type in ("gcn", "gin"):
        xr = x.clone().requires_grad_(True)
        out = model(xr, edge_index, batch)
        scalar = out if out.dim() == 0 else out.reshape(-1)[0]
        grad = torch.autograd.grad(scalar, xr)[0].detach().cpu().numpy()
        return np.linalg.norm(grad, axis=1)
    # gat: mean received attention across heads/layers
    with torch.no_grad():
        _, attn = model(x, edge_index, batch, return_attn=True)
    scores = np.zeros(x.shape[0])
    for ei, aw in attn:
        tg = ei[1].cpu().numpy(); w = aw.cpu().numpy().mean(axis=1)
        for t, wv in zip(tg, w):
            scores[t] += wv
    return scores / 3.0


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1: Crippen atomic-logP correlation
# ══════════════════════════════════════════════════════════════════════════════
def run_crippen(model, model_type, items, device, min_atoms=5):
    rhos = []
    for mol, g, smiles in items:
        n = mol.GetNumAtoms()
        if n < min_atoms:
            continue
        attr = atom_attribution(model, model_type, mol, g, smiles, device)
        contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        crippen = np.array([abs(c[0]) for c in contribs])   # |atomic logP contribution|
        if len(attr) != n or np.allclose(attr, attr[0]) or np.allclose(crippen, crippen[0]):
            continue
        rho = spearmanr(attr, crippen).correlation
        if not np.isnan(rho):
            rhos.append(rho)
    rhos = np.array(rhos)
    return {
        "n_molecules": int(len(rhos)),
        "mean_spearman": float(np.mean(rhos)) if len(rhos) else None,
        "median_spearman": float(np.median(rhos)) if len(rhos) else None,
        "sd_spearman": float(np.std(rhos)) if len(rhos) else None,
        "frac_positive": float(np.mean(rhos > 0)) if len(rhos) else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 2: structural-alert (toxicophore) enrichment
# ══════════════════════════════════════════════════════════════════════════════
def build_alert_catalog():
    p = FilterCatalogParams()
    p.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(p)


def alert_atoms(catalog, mol):
    """Set of atom indices belonging to any matched alert substructure."""
    atoms = set()
    for entry in catalog.GetMatches(mol):
        for fm in entry.GetFilterMatches(mol):
            for pair in fm.atomPairs:      # IntPair(queryIdx, molIdx)
                atoms.add(int(pair[1]))
    return atoms


def run_alerts(model, model_type, items, device, topk_frac=MASK_TOPK):
    catalog = build_alert_catalog()
    precisions, enrichments = [], []
    n_with_alert = 0
    for mol, g, smiles in items:
        n = mol.GetNumAtoms()
        alerts = alert_atoms(catalog, mol)
        if len(alerts) == 0 or len(alerts) == n:
            continue
        n_with_alert += 1
        attr = atom_attribution(model, model_type, mol, g, smiles, device)
        if len(attr) != n:
            continue
        k = max(1, int(n * topk_frac))
        top = set(np.argsort(attr)[::-1][:k].tolist())
        prec = len(top & alerts) / k                 # precision@k
        base = len(alerts) / n                       # overall alert-atom fraction
        precisions.append(prec)
        enrichments.append(prec / base if base > 0 else np.nan)
    precisions = np.array(precisions); enrichments = np.array(enrichments)
    enrichments = enrichments[~np.isnan(enrichments)]
    return {
        "n_molecules_with_alert": int(n_with_alert),
        "mean_precision_at_k": float(np.mean(precisions)) if len(precisions) else None,
        "mean_enrichment": float(np.mean(enrichments)) if len(enrichments) else None,
        "median_enrichment": float(np.median(enrichments)) if len(enrichments) else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True, choices=["crippen", "alerts"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=["ecfp4", "gcn", "gat", "gin"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--result-dir", default="results")
    args = ap.parse_args()

    device = torch.device("cpu")
    model, items, task = load_model_and_data(args.dataset, args.model, args.data_dir,
                                             args.result_dir, device)
    print(f"\n{'='*55}")
    print(f"Chem validation [{args.analysis}]: {args.dataset} | {args.model.upper()} | "
          f"{len(items)} molecules")

    if args.analysis == "crippen":
        res = run_crippen(model, args.model, items, device)
        print(f"  mean Spearman(attr, |Crippen|) = {res['mean_spearman']}  "
              f"(n={res['n_molecules']}, frac+ = {res['frac_positive']})")
    else:
        res = run_alerts(model, args.model, items, device)
        print(f"  precision@k = {res['mean_precision_at_k']}  "
              f"enrichment = {res['mean_enrichment']}  "
              f"(n_alert_mols = {res['n_molecules_with_alert']})")

    res.update({"analysis": args.analysis, "dataset": args.dataset,
                "model": args.model, "task": task})
    out_dir = Path(args.result_dir) / "chem_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.analysis}_{args.dataset}_{args.model}.json", "w") as f:
        json.dump(res, f, indent=2)

    summ = Path(args.result_dir) / "chem_validation_summary.csv"
    row = pd.DataFrame([res])
    if summ.exists():
        old = pd.read_csv(summ)
        old = old[~((old["analysis"] == args.analysis) & (old["dataset"] == args.dataset)
                    & (old["model"] == args.model))]
        pd.concat([old, row], ignore_index=True).to_csv(summ, index=False)
    else:
        row.to_csv(summ, index=False)
    print(f"  Saved → {out_dir}/ and {summ}")


if __name__ == "__main__":
    main()
