"""
08_xai_baselines_bootstrap.py
==============================
JCIM 리뷰 대응: 기존 native saliency(ECFP4/GCN gradient, GAT attention) 외에
model-agnostic한 Integrated Gradients(IG)와 Grad×Input(GxI)을 추가 attribution
방법으로 계산하고, 06_bootstrap_ci.py와 동일한 masking + bootstrap 파이프라인으로
ACCS_vs_perm / ACCS_vs_matched를 재보고한다.

핵심 목적:
  - "molecular XAI 선행연구(deletion/fidelity) 대비 새 baseline이 부족하다"는 지적에
    대응해 최소 2개의 널리 쓰이는 attribution baseline을 추가.
  - GAT의 경우 특히: "raw attention이 아니라 이 GAT 모델에 gradient 기반 설명을
    적용하면 어떻게 되는가"를 직접 비교 가능하게 함 (attention 자체의 문제인지,
    모델 전반의 문제인지 분리).

전제 조건: 02_train_models.py / 04_train_gnn.py 로 모델이 이미 학습되어 있어야 함
  (이 스크립트는 저장된 best_model.pt를 그대로 불러와 saliency만 다시 계산).

실행:
  cd /path/to/AttnBench
  python experiments/08_xai_baselines_bootstrap.py --dataset BACE --model gcn --method ig
  python experiments/08_xai_baselines_bootstrap.py --dataset BACE --model gcn --method gxi
  python experiments/08_xai_baselines_bootstrap.py --dataset BACE --model gat --method ig --seed 0

  # 전체 조합 (7 datasets x 3 models x 2 methods = 42회):
  python -c "
  import subprocess, sys
  datasets = ['BBBP','BACE','HIV','ClinTox','ESOL','Lipophilicity','FreeSolv']
  for ds in datasets:
      for m in ['ecfp4','gcn','gat']:
          for method in ['ig','gxi']:
              subprocess.run([sys.executable, 'experiments/08_xai_baselines_bootstrap.py',
                              '--dataset', ds, '--model', m, '--method', method])
  " | tee logs/xai_baselines.log

출력:
  results/<dataset>/<model_subdir>/xai_{method}/bootstrap_ci.json          (seed=42, ratio=0.2)
  results/<dataset>/<model_subdir>/xai_{method}/seed_{N}/bootstrap_ci.json (seed=N, N != 42)
  results/xai_bootstrap_summary.csv                                        (seed=42, ratio=0.2 누적)

09_multiseed_summary.py --include-xai 로 native 결과와 함께 BH 보정 집계 가능.
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

# ─── 06_bootstrap_ci.py 재사용 (숫자로 시작하는 파일명은 일반 import 불가 → 동적 로드) ──
_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("bootstrap_ci_06", _HERE / "06_bootstrap_ci.py")
bci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bci)

sys.path.insert(0, str(_HERE))

N_STEPS_DEFAULT = 50  # Integrated Gradients 적분 스텝 수


# ══════════════════════════════════════════════════════════════════════════════
# Integrated Gradients / Grad×Input — ECFP4 (fingerprint bit 입력)
# ══════════════════════════════════════════════════════════════════════════════
def ig_ecfp4_atom_saliency(model, fp_arr, bi, n_atoms, device, steps=N_STEPS_DEFAULT):
    """IG: baseline(전부 0인 fingerprint) → 실제 fingerprint 경로를 따라 gradient를
    적분(리만 합)하고 (x - baseline)을 곱한다. Bit → atom 매핑은 기존과 동일하게
    |IG| 합산."""
    fp_t   = torch.tensor(fp_arr, dtype=torch.float32, device=device)
    base_t = torch.zeros_like(fp_t)
    alphas = np.linspace(0.0, 1.0, steps + 1)[1:]  # alpha=0은 gradient가 아니라 baseline 자체이므로 제외

    total_grad = torch.zeros_like(fp_t)
    for alpha in alphas:
        x = (base_t + float(alpha) * (fp_t - base_t)).clone().requires_grad_(True)
        out = model(x)
        scalar = out if out.dim() == 0 else out[0]
        grad = torch.autograd.grad(scalar, x)[0]
        total_grad = total_grad + grad.detach()
    avg_grad = total_grad / steps
    ig = ((fp_t - base_t) * avg_grad).cpu().numpy()

    atom_scores = np.zeros(n_atoms)
    for bit, envs in bi.items():
        contrib = abs(float(ig[bit]))
        for (atom_idx, _) in envs:
            if atom_idx < n_atoms:
                atom_scores[atom_idx] += contrib
    return atom_scores


def gxi_ecfp4_atom_saliency(model, fp_arr, bi, n_atoms, device):
    """Grad x Input: 입력 fingerprint에서의 gradient에 입력값 자체를 곱함."""
    x = torch.tensor(fp_arr, dtype=torch.float32, device=device, requires_grad=True)
    out = model(x)
    scalar = out if out.dim() == 0 else out[0]
    grad = torch.autograd.grad(scalar, x)[0].detach()
    gxi = (grad * x.detach()).cpu().numpy()

    atom_scores = np.zeros(n_atoms)
    for bit, envs in bi.items():
        contrib = abs(float(gxi[bit]))
        for (atom_idx, _) in envs:
            if atom_idx < n_atoms:
                atom_scores[atom_idx] += contrib
    return atom_scores


# ══════════════════════════════════════════════════════════════════════════════
# Integrated Gradients / Grad×Input — GCN/GAT (node feature 입력, model-agnostic)
# ══════════════════════════════════════════════════════════════════════════════
def ig_node_saliency(model, x, edge_index, batch, device, steps=N_STEPS_DEFAULT):
    """GCN/GAT 공통: node feature에 대해 IG 계산 후 원자별 L2 norm으로 축약.
    GAT의 경우 attention coefficient를 읽지 않고, 이 GAT 모델의 순수 gradient
    신호만 사용한다 — "attention 자체가 문제인지, 모델 전체가 문제인지" 분리 목적."""
    x = x.to(device)
    edge_index = edge_index.to(device)
    base = torch.zeros_like(x)
    alphas = np.linspace(0.0, 1.0, steps + 1)[1:]

    total_grad = torch.zeros_like(x)
    for alpha in alphas:
        xi = (base + float(alpha) * (x - base)).clone().detach().requires_grad_(True)
        out = model(xi, edge_index, batch)
        scalar = out if out.dim() == 0 else out[0]
        grad = torch.autograd.grad(scalar, xi)[0]
        total_grad = total_grad + grad.detach()
    avg_grad = total_grad / steps
    ig = ((x - base) * avg_grad).cpu().numpy()
    return np.linalg.norm(ig, axis=1)


def gxi_node_saliency(model, x, edge_index, batch, device):
    xi = x.clone().detach().to(device).requires_grad_(True)
    edge_index = edge_index.to(device)
    out = model(xi, edge_index, batch)
    scalar = out if out.dim() == 0 else out[0]
    grad = torch.autograd.grad(scalar, xi)[0].detach()
    gxi = (grad * xi.detach()).cpu().numpy()
    return np.linalg.norm(gxi, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# ECFP4-MLP: XAI 방법으로 bootstrap
# ══════════════════════════════════════════════════════════════════════════════
def run_ecfp4_xai_bootstrap(dataset, task, data_dir, result_dir, device, method, seed, n_steps):
    data_path, model_dir = bci.resolve_seed_dirs(dataset, data_dir, result_dir, "mlp", seed)
    model_path = model_dir / "best_model.pt"
    if not model_path.exists():
        print(f"  [SKIP] No ECFP4 model: {model_path}"); return None

    model = bci.MLP().to(device)
    # 02_train_models.py 저장 형식({"state_dict":...} 로 감싼 dict)에 맞춰 언랩 (06과 동일한 수정).
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    test_df = pd.read_csv(data_path / "test.csv")
    mols = []
    for _, row in test_df.iterrows():
        fp, bi = bci.smiles_to_ecfp4(row["smiles"])
        if fp is None:
            continue
        mol = Chem.MolFromSmiles(row["smiles"])
        n_atoms = mol.GetNumAtoms()
        mols.append({"fp": fp, "bi": bi, "n_atoms": n_atoms, "mol": mol,
                     "label": row[bci.DATASETS[dataset]["label_col"]]})

    if not mols:
        print("  [SKIP] No valid molecules."); return None

    def saliency(m):
        if method == "ig":
            return ig_ecfp4_atom_saliency(model, m["fp"], m["bi"], m["n_atoms"], device, steps=n_steps)
        else:
            return gxi_ecfp4_atom_saliency(model, m["fp"], m["bi"], m["n_atoms"], device)

    labels = np.array([m["label"] for m in mols])
    k_per_mol = [max(1, int(m["n_atoms"] * bci.MASK_RATIO)) for m in mols]

    print(f"  Computing {method.upper()} saliency for {len(mols)} molecules...", flush=True)
    saliencies = [saliency(m) for m in mols]

    base_preds = []
    for m in mols:
        with torch.no_grad():
            base_preds.append(model(torch.tensor(m["fp"]).to(device)).item())
    base_preds = np.array(base_preds)

    true_preds, true_top_atoms = [], []
    for i, m in enumerate(mols):
        k = k_per_mol[i]
        top_atoms = np.argsort(saliencies[i])[::-1][:k].copy()
        true_top_atoms.append(top_atoms)
        masked_fp = bci.mask_ecfp4(m["fp"], top_atoms, m["bi"])
        with torch.no_grad():
            true_preds.append(model(torch.tensor(masked_fp).to(device)).item())
    true_preds = np.array(true_preds)

    perm_preds_all = []
    for _ in range(bci.N_REPEAT):
        pp = []
        for i, m in enumerate(mols):
            k = k_per_mol[i]
            perm_sal = np.random.permutation(saliencies[i])
            top_atoms = np.argsort(perm_sal)[::-1][:k].copy()
            masked_fp = bci.mask_ecfp4(m["fp"], top_atoms, m["bi"])
            with torch.no_grad():
                pp.append(model(torch.tensor(masked_fp).to(device)).item())
        perm_preds_all.append(np.array(pp))

    rand_preds_all = []
    for _ in range(bci.N_REPEAT):
        rp = []
        for i, m in enumerate(mols):
            k = k_per_mol[i]
            rand_atoms = np.random.choice(m["n_atoms"], k, replace=False)
            masked_fp = bci.mask_ecfp4(m["fp"], rand_atoms, m["bi"])
            with torch.no_grad():
                rp.append(model(torch.tensor(masked_fp).to(device)).item())
        rand_preds_all.append(np.array(rp))

    matched_preds_all = []
    for _ in range(bci.N_REPEAT):
        mp = []
        for i, m in enumerate(mols):
            k = k_per_mol[i]
            matched_atoms = bci.degree_matched_atoms(m["mol"], true_top_atoms[i], k)
            masked_fp = bci.mask_ecfp4(m["fp"], matched_atoms, m["bi"])
            with torch.no_grad():
                mp.append(model(torch.tensor(masked_fp).to(device)).item())
        matched_preds_all.append(np.array(mp))

    return labels, base_preds, true_preds, perm_preds_all, rand_preds_all, matched_preds_all, task


# ══════════════════════════════════════════════════════════════════════════════
# GCN/GAT: XAI 방법으로 bootstrap
# ══════════════════════════════════════════════════════════════════════════════
def run_gnn_xai_bootstrap(dataset, model_type, task, data_dir, result_dir, device, method, seed, n_steps):
    try:
        from train_gnn_utils import (
            DATASETS as GNN_DATASETS, smiles_to_data, GCNModel, GATModel, GINModel
        )
    except ImportError as e:
        print(f"  [SKIP] {e}"); return None

    base_subdir = f"gnn_{model_type}"
    data_path, model_dir = bci.resolve_seed_dirs(dataset, data_dir, result_dir, base_subdir, seed)
    model_path = model_dir / "best_model.pt"
    if not model_path.exists():
        print(f"  [SKIP] No model: {model_path}"); return None

    test_df = pd.read_csv(data_path / "test.csv")
    meta = GNN_DATASETS[dataset]

    test_graphs, test_mols = [], []
    for _, row in test_df.iterrows():
        g = smiles_to_data(row["smiles"], row[meta["label_col"]], task)
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if g is not None and mol is not None and mol.GetNumAtoms() == g.x.shape[0]:
            test_graphs.append(g)
            test_mols.append(mol)

    if not test_graphs:
        print("  [SKIP] No test graphs."); return None

    in_dim = test_graphs[0].x.shape[1]
    if model_type == "gcn":
        model = GCNModel(in_dim)
    elif model_type == "gin":
        model = GINModel(in_dim)
    else:
        model = GATModel(in_dim, hidden_dim=64, heads=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    labels = np.array([g.y.numpy().item() for g in test_graphs])

    def saliency(g):
        batch = torch.zeros(g.x.shape[0], dtype=torch.long, device=device)
        if method == "ig":
            return ig_node_saliency(model, g.x, g.edge_index, batch, device, steps=n_steps)
        else:
            return gxi_node_saliency(model, g.x, g.edge_index, batch, device)

    def masked_pred(g, mask_atoms):
        x_masked = g.x.clone()
        if len(mask_atoms) > 0:
            x_masked[np.array(mask_atoms)] = 0.0
        batch = torch.zeros(x_masked.shape[0], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(x_masked.to(device), g.edge_index.to(device), batch)
        return out.item()

    print(f"  Computing {method.upper()} saliency for {len(test_graphs)} graphs...", flush=True)
    saliencies = [saliency(g) for g in test_graphs]
    k_per_mol = [max(1, int(g.x.shape[0] * bci.MASK_RATIO)) for g in test_graphs]

    base_preds = np.array([masked_pred(g, []) for g in test_graphs])

    true_preds, true_top_atoms = [], []
    for i, (g, sal) in enumerate(zip(test_graphs, saliencies)):
        k = k_per_mol[i]
        top_atoms = np.argsort(sal)[::-1][:k].copy()
        true_top_atoms.append(top_atoms)
        true_preds.append(masked_pred(g, top_atoms))
    true_preds = np.array(true_preds)

    perm_preds_all = []
    for _ in range(bci.N_REPEAT):
        pp = []
        for i, (g, sal) in enumerate(zip(test_graphs, saliencies)):
            k = k_per_mol[i]
            perm_sal = np.random.permutation(sal)
            top_atoms = np.argsort(perm_sal)[::-1][:k].copy()
            pp.append(masked_pred(g, top_atoms))
        perm_preds_all.append(np.array(pp))

    rand_preds_all = []
    for _ in range(bci.N_REPEAT):
        rp = []
        for i, g in enumerate(test_graphs):
            k = k_per_mol[i]
            rand_atoms = np.random.choice(g.x.shape[0], k, replace=False)
            rp.append(masked_pred(g, rand_atoms))
        rand_preds_all.append(np.array(rp))

    matched_preds_all = []
    for _ in range(bci.N_REPEAT):
        mp = []
        for i, g in enumerate(test_graphs):
            k = k_per_mol[i]
            matched_atoms = bci.degree_matched_atoms(test_mols[i], true_top_atoms[i], k)
            mp.append(masked_pred(g, matched_atoms))
        matched_preds_all.append(np.array(mp))

    return labels, base_preds, true_preds, perm_preds_all, rand_preds_all, matched_preds_all, task


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--model",      required=True, choices=["ecfp4", "gcn", "gat", "gin"])
    parser.add_argument("--method",     required=True, choices=["ig", "gxi"],
                         help="ig=Integrated Gradients, gxi=Grad x Input")
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--n-bootstrap", type=int, default=bci.N_BOOTSTRAP)
    parser.add_argument("--n-steps",    type=int, default=N_STEPS_DEFAULT,
                         help="Integrated Gradients 적분 스텝 수 (gxi에는 미사용)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--mask-ratio", type=float, default=bci.MASK_RATIO)
    args = parser.parse_args()
    bci.MASK_RATIO = args.mask_ratio  # bci 모듈 내부 상수를 공유해서 사용

    if args.dataset not in bci.DATASETS:
        print(f"Unknown dataset: {args.dataset}"); return

    task       = bci.DATASETS[args.dataset]["task"]
    data_dir   = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    device     = torch.device("cpu")  # bootstrap은 CPU로 충분 (기존 06과 동일)

    base_subdir = "mlp" if args.model == "ecfp4" else f"gnn_{args.model}"
    xai_subdir  = f"{base_subdir}/xai_{args.method}"
    _, out_dir  = bci.resolve_seed_dirs(args.dataset, data_dir, result_dir, xai_subdir, args.seed)

    print(f"\n{'='*55}")
    print(f"XAI Bootstrap: {args.dataset} | {args.model.upper()} | method={args.method} | "
          f"seed={args.seed} | mask_ratio={bci.MASK_RATIO} | n_bootstrap={args.n_bootstrap}")

    if args.model == "ecfp4":
        result = run_ecfp4_xai_bootstrap(args.dataset, task, data_dir, result_dir, device,
                                          args.method, args.seed, args.n_steps)
    else:
        result = run_gnn_xai_bootstrap(args.dataset, args.model, task, data_dir, result_dir, device,
                                        args.method, args.seed, args.n_steps)

    if result is None:
        return

    labels, base_preds, true_preds, perm_preds_all, rand_preds_all, matched_preds_all, task = result
    print(f"  n_test={len(labels)}")

    perf_base    = bci.compute_metric(base_preds, labels, task)
    perf_true    = bci.compute_metric(true_preds, labels, task)
    perf_perm    = np.mean([bci.compute_metric(p, labels, task) for p in perm_preds_all])
    perf_matched = np.mean([bci.compute_metric(m, labels, task) for m in matched_preds_all])
    obs_accs_vs_perm    = (perf_base - perf_true) - (perf_base - perf_perm)
    obs_accs_vs_matched = (perf_base - perf_true) - (perf_base - perf_matched)

    print(f"  Observed ACCS_vs_perm    = {obs_accs_vs_perm:+.4f}")
    print(f"  Observed ACCS_vs_matched = {obs_accs_vs_matched:+.4f}")

    print(f"  Running {args.n_bootstrap} bootstrap iterations...", flush=True)
    accs_s, accs_perm_s, accs_vs_perm_s, accs_matched_s, accs_vs_matched_s = bci.run_bootstrap(
        labels, base_preds, true_preds, perm_preds_all, rand_preds_all, matched_preds_all,
        task, args.n_bootstrap
    )

    mean_avp, lo_avp, hi_avp, p_avp = bci.bootstrap_ci(accs_vs_perm_s)
    mean_a,   lo_a,   hi_a,   p_a   = bci.bootstrap_ci(accs_s)
    mean_avm, lo_avm, hi_avm, p_avm = bci.bootstrap_ci(accs_vs_matched_s)

    def stars(p):
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    sig, sig_avm = stars(p_avp), stars(p_avm)
    print(f"  ACCS_vs_perm    = {mean_avp:+.4f} [{lo_avp:+.4f}, {hi_avp:+.4f}]  p={p_avp:.4f} {sig}")
    print(f"  ACCS_vs_matched = {mean_avm:+.4f} [{lo_avm:+.4f}, {hi_avm:+.4f}]  p={p_avm:.4f} {sig_avm}")

    ci_result = {
        "dataset": args.dataset, "model": args.model, "method": args.method,
        "task": task, "seed": args.seed,
        "n_test": int(len(labels)), "n_bootstrap": args.n_bootstrap,
        "mask_ratio": bci.MASK_RATIO, "n_steps": args.n_steps,
        "obs_accs_vs_perm": float(obs_accs_vs_perm),
        "bootstrap_mean": float(mean_avp),
        "ci_lo": float(lo_avp), "ci_hi": float(hi_avp),
        "p_value": float(p_avp), "significant": sig,
        "ACCS_mean": float(mean_a), "ACCS_ci_lo": float(lo_a), "ACCS_ci_hi": float(hi_a),
        "obs_accs_vs_matched": float(obs_accs_vs_matched),
        "matched_bootstrap_mean": float(mean_avm),
        "matched_ci_lo": float(lo_avm), "matched_ci_hi": float(hi_avm),
        "matched_p_value": float(p_avm), "matched_significant": sig_avm,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ratio_tag = "" if abs(bci.MASK_RATIO - 0.2) < 1e-9 else f"_ratio{bci.MASK_RATIO}"
    ci_filename = f"bootstrap_ci{ratio_tag}.json"
    with open(out_dir / ci_filename, "w") as f:
        json.dump(ci_result, f, indent=2)

    if args.seed == 42 and ratio_tag == "":
        summary_path = result_dir / "xai_bootstrap_summary.csv"
        new_row = pd.DataFrame([ci_result])
        if summary_path.exists():
            old = pd.read_csv(summary_path)
            old = old[~((old["dataset"] == args.dataset) & (old["model"] == args.model)
                        & (old["method"] == args.method))]
            pd.concat([old, new_row], ignore_index=True).to_csv(summary_path, index=False)
        else:
            new_row.to_csv(summary_path, index=False)

    print(f"  Saved: {out_dir}/{ci_filename}")


if __name__ == "__main__":
    main()
