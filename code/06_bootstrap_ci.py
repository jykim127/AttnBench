"""
06_bootstrap_ci.py
===================
Bootstrap CI (B=1000) for ACCS_vs_perm.

핵심 아이디어:
  1. 각 분자별 masked prediction을 미리 저장
     (baseline / true / perm×N_REPEAT / rand×N_REPEAT / degree_matched×N_REPEAT)
  2. Bootstrap: 분자 index를 replacement로 재샘플링 → metric 재계산
  3. 95% CI + one-sided p-value (H0: ACCS_vs_perm ≤ 0)

지원 모델: ecfp4 / gcn / gat

JCIM 리뷰 대응으로 추가된 것 (기존 --dataset/--model 실행법은 그대로 동작):
  --seed N        : scaffold split seed (01_download_data.py --seeds 로 미리 생성해둔
                     data/<ds>/split_seed{N}/ 을 사용). 기본 42 = 기존 결과와 동일 경로.
  --mask-ratio R  : top-k 비율 (기본 0.2). 0.1/0.2/0.3 스윕으로 SI 표를 만들 때 사용.
  degree-matched baseline : permuted/random 외에, true top-k와 동일한 atom-degree
                     분포를 갖도록 다른 원자로 치환한 대조군을 추가. "permuted가 사실상
                     random과 다를 게 없다"는 지적에 대한 더 엄격한 대조군.

실행:
  cd /path/to/AttnBench
  # 단일 (기존과 동일):
  python experiments/06_bootstrap_ci.py --dataset BACE --model gcn
  # multi-seed:
  python experiments/06_bootstrap_ci.py --dataset BACE --model gcn --seed 0
  # mask-ratio 스윕:
  python experiments/06_bootstrap_ci.py --dataset BACE --model gcn --mask-ratio 0.1
  python experiments/06_bootstrap_ci.py --dataset BACE --model gcn --mask-ratio 0.3
  # 전체:
  python -c "
  import subprocess, sys
  datasets = ['BBBP','BACE','HIV','ClinTox','ESOL','Lipophilicity','FreeSolv']
  for ds in datasets:
      for m in ['ecfp4','gcn','gat']:
          print(f'=== {ds} {m} ===', flush=True)
          subprocess.run([sys.executable, 'experiments/06_bootstrap_ci.py', '--dataset', ds, '--model', m])
  " | tee logs/bootstrap.log

출력:
  results/<dataset>/<model>/bootstrap_ci.json                                  (seed=42, ratio=0.2 기본 경로 그대로)
  results/<dataset>/<model>/seed_{N}/bootstrap_ci_ratio{R}.json                (seed/ratio 지정 시)
  results/bootstrap_summary.csv                                                (seed=42, ratio=0.2 누적)

이후 09_multiseed_summary.py 로 seed 간 집계 + Benjamini-Hochberg 다중검정 보정 수행.
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, mean_squared_error

# ─── 설정 ─────────────────────────────────────────────────────────────────────
N_REPEAT    = 20    # perm/rand/degree_matched 반복
N_BOOTSTRAP = 1000  # bootstrap 반복
MASK_RATIO  = 0.2   # 논문 메인 결과와 동일 (--mask-ratio 로 오버라이드 가능)
SEED        = 42


# ─── Degree-matched masking baseline ──────────────────────────────────────────
def degree_matched_atoms(mol, true_atoms, k):
    """true top-k 원자와 동일한 atom-degree 분포를 갖도록, 서로 다른 원자로 치환한
    top-k 집합을 만든다 (permuted saliency보다 엄격한 대조군).

    각 true 원자에 대해: (1) 동일 degree를 가진 미사용 원자 중 무작위 선택,
    (2) 없으면 degree가 가장 가까운 미사용 원자, (3) 그래도 없으면 무작위로 채움.
    """
    n_atoms = mol.GetNumAtoms()
    degrees = np.array([mol.GetAtomWithIdx(i).GetDegree() for i in range(n_atoms)])
    true_set = set(int(a) for a in true_atoms)
    pool = [i for i in range(n_atoms) if i not in true_set]

    chosen, used = [], set()
    for a in true_atoms:
        d = degrees[int(a)]
        same_deg = [i for i in pool if i not in used and degrees[i] == d]
        if same_deg:
            pick = same_deg[np.random.randint(len(same_deg))]
        else:
            remaining = [i for i in pool if i not in used]
            if not remaining:
                continue
            remaining = sorted(remaining, key=lambda i: abs(int(degrees[i]) - int(d)))
            pick = remaining[0]
        chosen.append(pick)
        used.add(pick)

    if len(chosen) < k:
        remaining = [i for i in pool if i not in used]
        np.random.shuffle(remaining)
        chosen += remaining[:k - len(chosen)]

    return np.array(chosen[:k], dtype=int)

DATASETS = {
    "BBBP":         {"task": "classification", "label_col": "label"},
    "BACE":         {"task": "classification", "label_col": "label"},
    "HIV":          {"task": "classification", "label_col": "label"},
    "ClinTox":      {"task": "classification", "label_col": "label"},
    "ESOL":         {"task": "regression",     "label_col": "label"},
    "Lipophilicity":{"task": "regression",     "label_col": "label"},
    "FreeSolv":     {"task": "regression",     "label_col": "label"},
}


# ─── 공통 metric 함수 ──────────────────────────────────────────────────────────
def compute_metric(preds, labels, task):
    if task == "classification":
        probs = 1 / (1 + np.exp(-np.array(preds, dtype=float)))
        probs = np.nan_to_num(probs, nan=0.5)
        try:
            return roc_auc_score(labels, probs)
        except Exception:
            return 0.5
    else:
        rmse = np.sqrt(mean_squared_error(labels, np.array(preds, dtype=float)))
        return -rmse


def bootstrap_ci(accs_vs_perm_samples, alpha=0.05):
    """Bootstrap mean, 95% CI, one-sided p-value."""
    arr = np.array(accs_vs_perm_samples)
    mean = float(arr.mean())
    ci_lo = float(np.percentile(arr, 100 * alpha / 2))
    ci_hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    # one-sided p-value: proportion of bootstrap samples ≤ 0
    p_val = float((arr <= 0).mean())
    return mean, ci_lo, ci_hi, p_val


# ══════════════════════════════════════════════════════════════════════════════
# ECFP4 섹션
# ══════════════════════════════════════════════════════════════════════════════
class MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden1=512, hidden2=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def smiles_to_ecfp4(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, bitInfo=bi)
    arr = np.zeros(nbits, dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr, bi


def ecfp4_atom_saliency(model, fp_tensor, bi, n_atoms, device):
    """ECFP4 gradient saliency → atom score."""
    x = fp_tensor.clone().requires_grad_(True).to(device)
    out = model(x)
    scalar = out if out.dim() == 0 else out[0]
    grad = torch.autograd.grad(scalar, x)[0].detach().cpu().numpy()

    atom_scores = np.zeros(n_atoms)
    for bit, envs in bi.items():
        g = abs(grad[bit])
        for (atom_idx, _) in envs:
            if atom_idx < n_atoms:
                atom_scores[atom_idx] += g
    return atom_scores


def mask_ecfp4(fp_arr, atom_indices, bi):
    """특정 원자에 해당하는 ECFP4 bit를 0으로."""
    masked = fp_arr.copy()
    atom_set = set(atom_indices)
    for bit, envs in bi.items():
        if any(a in atom_set for (a, _) in envs):
            masked[bit] = 0.0
    return masked


def resolve_seed_dirs(dataset, data_dir, result_dir, model_subdir, seed):
    """seed=42는 기존 경로 그대로, 그 외는 split_seed{N}/, seed_{N}/ 하위 사용."""
    data_path = data_dir / dataset if seed == 42 else data_dir / dataset / f"split_seed{seed}"
    model_dir = (result_dir / dataset / model_subdir if seed == 42
                 else result_dir / dataset / model_subdir / f"seed_{seed}")
    return data_path, model_dir


def run_ecfp4_bootstrap(dataset, task, data_dir, result_dir, device, seed=42):
    """ECFP4-MLP bootstrap CI 계산."""
    data_path, model_dir = resolve_seed_dirs(dataset, data_dir, result_dir, "mlp", seed)
    model_path = model_dir / "best_model.pt"
    if not model_path.exists():
        print(f"  [SKIP] No ECFP4 model: {model_path}"); return None

    model = MLP().to(device)
    # 02_train_models.py는 raw state_dict가 아니라
    # {"state_dict":..., "out_dim":..., "task":..., "metrics":...} 로 감싸서 저장한다.
    # (예전 코드에서 이 부분이 raw state_dict를 기대하고 있어서 항상 로딩에 실패하던 버그)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    test_df = pd.read_csv(data_path / "test.csv")
    mols = []
    for _, row in test_df.iterrows():
        fp, bi = smiles_to_ecfp4(row["smiles"])
        if fp is None:
            continue
        mol = Chem.MolFromSmiles(row["smiles"])
        n_atoms = mol.GetNumAtoms()
        mols.append({"fp": fp, "bi": bi, "n_atoms": n_atoms, "mol": mol,
                     "label": row[DATASETS[dataset]["label_col"]]})

    if not mols:
        print("  [SKIP] No valid molecules."); return None

    labels = np.array([m["label"] for m in mols])
    n = len(mols)
    k_per_mol = [max(1, int(m["n_atoms"] * MASK_RATIO)) for m in mols]

    # ── 1. 기준 예측 ──────────────────────────────────────────────────────────
    base_preds = []
    for m in mols:
        with torch.no_grad():
            x = torch.tensor(m["fp"]).to(device)
            base_preds.append(model(x).item())
    base_preds = np.array(base_preds)

    # ── 2. True saliency masking ──────────────────────────────────────────────
    true_preds = []
    true_top_atoms = []  # degree-matched baseline에서 재사용
    for m in mols:
        x_t = torch.tensor(m["fp"]).to(device)
        sal = ecfp4_atom_saliency(model, x_t, m["bi"], m["n_atoms"], device)
        k = max(1, int(m["n_atoms"] * MASK_RATIO))
        top_atoms = np.argsort(sal)[::-1][:k].copy()
        true_top_atoms.append(top_atoms)
        masked_fp = mask_ecfp4(m["fp"], top_atoms, m["bi"])
        with torch.no_grad():
            pred = model(torch.tensor(masked_fp).to(device)).item()
        true_preds.append(pred)
    true_preds = np.array(true_preds)

    # ── 3. Permuted saliency masking (N_REPEAT) ───────────────────────────────
    perm_preds_all = []
    for _ in range(N_REPEAT):
        pp = []
        for m in mols:
            x_t = torch.tensor(m["fp"]).to(device)
            sal = ecfp4_atom_saliency(model, x_t, m["bi"], m["n_atoms"], device)
            perm_sal = np.random.permutation(sal)
            k = max(1, int(m["n_atoms"] * MASK_RATIO))
            top_atoms = np.argsort(perm_sal)[::-1][:k].copy()
            masked_fp = mask_ecfp4(m["fp"], top_atoms, m["bi"])
            with torch.no_grad():
                pred = model(torch.tensor(masked_fp).to(device)).item()
            pp.append(pred)
        perm_preds_all.append(np.array(pp))

    # ── 4. Random masking (N_REPEAT) ──────────────────────────────────────────
    rand_preds_all = []
    for _ in range(N_REPEAT):
        rp = []
        for i, m in enumerate(mols):
            k = k_per_mol[i]
            rand_atoms = np.random.choice(m["n_atoms"], k, replace=False)
            masked_fp = mask_ecfp4(m["fp"], rand_atoms, m["bi"])
            with torch.no_grad():
                pred = model(torch.tensor(masked_fp).to(device)).item()
            rp.append(pred)
        rand_preds_all.append(np.array(rp))

    # ── 5. Degree-matched masking (N_REPEAT) ──────────────────────────────────
    # true top-k와 동일한 atom-degree 분포를 갖는 다른 원자로 치환한 대조군.
    # permuted가 random과 사실상 구분 안 된다는 지적에 대응하는 더 엄격한 baseline.
    matched_preds_all = []
    for _ in range(N_REPEAT):
        mp = []
        for i, m in enumerate(mols):
            k = k_per_mol[i]
            matched_atoms = degree_matched_atoms(m["mol"], true_top_atoms[i], k)
            masked_fp = mask_ecfp4(m["fp"], matched_atoms, m["bi"])
            with torch.no_grad():
                pred = model(torch.tensor(masked_fp).to(device)).item()
            mp.append(pred)
        matched_preds_all.append(np.array(mp))

    return labels, base_preds, true_preds, perm_preds_all, rand_preds_all, matched_preds_all, task


# ══════════════════════════════════════════════════════════════════════════════
# GNN 섹션
# ══════════════════════════════════════════════════════════════════════════════
def run_gnn_bootstrap(dataset, model_type, task, data_dir, result_dir, device, seed=42):
    """GCN/GAT bootstrap CI 계산."""
    try:
        from torch_geometric.data import Data
        from train_gnn_utils import (
            DATASETS as GNN_DATASETS, smiles_to_data, GCNModel, GATModel, GINModel
        )
    except ImportError as e:
        print(f"  [SKIP] {e}"); return None

    data_path, model_dir = resolve_seed_dirs(dataset, data_dir, result_dir, f"gnn_{model_type}", seed)
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
    n = len(test_graphs)

    def get_saliency(g):
        if model_type in ("gcn", "gin"):
            x = g.x.clone().requires_grad_(True).to(device)
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            out = model(x, g.edge_index.to(device), batch)
            scalar = out if out.dim() == 0 else out[0]
            grad = torch.autograd.grad(scalar, x)[0].detach().cpu().numpy()
            return np.linalg.norm(grad, axis=1)
        else:
            x = g.x.to(device)
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            with torch.no_grad():
                _, attn_tuple = model(x, g.edge_index.to(device), batch, return_attn=True)
            n_atoms = x.shape[0]
            scores = np.zeros(n_atoms)
            for (ei, aw) in attn_tuple:
                tgts = ei[1].cpu().numpy()
                wts  = aw.cpu().numpy().mean(axis=1)
                for t, w in zip(tgts, wts):
                    scores[t] += w
            return scores / 3.0

    def masked_pred(g, mask_atoms):
        x_masked = g.x.clone()
        if len(mask_atoms) > 0:
            x_masked[np.array(mask_atoms)] = 0.0
        batch = torch.zeros(x_masked.shape[0], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(x_masked.to(device), g.edge_index.to(device), batch)
        return out.item()

    # ── Saliency 계산 ──────────────────────────────────────────────────────────
    print("  Computing saliency...", flush=True)
    saliencies = [get_saliency(g) for g in test_graphs]
    k_per_mol = [max(1, int(g.x.shape[0] * MASK_RATIO)) for g in test_graphs]

    # ── Baseline predictions ───────────────────────────────────────────────────
    base_preds = np.array([masked_pred(g, []) for g in test_graphs])

    # ── True saliency masking ─────────────────────────────────────────────────
    true_preds = []
    true_top_atoms = []  # degree-matched baseline에서 재사용
    for i, (g, sal) in enumerate(zip(test_graphs, saliencies)):
        k = k_per_mol[i]
        top_atoms = np.argsort(sal)[::-1][:k].copy()
        true_top_atoms.append(top_atoms)
        true_preds.append(masked_pred(g, top_atoms))
    true_preds = np.array(true_preds)

    # ── Permuted saliency masking ─────────────────────────────────────────────
    perm_preds_all = []
    for _ in range(N_REPEAT):
        pp = []
        for i, (g, sal) in enumerate(zip(test_graphs, saliencies)):
            k = k_per_mol[i]
            perm_sal = np.random.permutation(sal)
            top_atoms = np.argsort(perm_sal)[::-1][:k].copy()
            pp.append(masked_pred(g, top_atoms))
        perm_preds_all.append(np.array(pp))

    # ── Random masking ────────────────────────────────────────────────────────
    rand_preds_all = []
    for _ in range(N_REPEAT):
        rp = []
        for i, g in enumerate(test_graphs):
            n_atoms = g.x.shape[0]
            k = k_per_mol[i]
            rand_atoms = np.random.choice(n_atoms, k, replace=False)
            rp.append(masked_pred(g, rand_atoms))
        rand_preds_all.append(np.array(rp))

    # ── Degree-matched masking ────────────────────────────────────────────────
    matched_preds_all = []
    for _ in range(N_REPEAT):
        mp = []
        for i, g in enumerate(test_graphs):
            k = k_per_mol[i]
            matched_atoms = degree_matched_atoms(test_mols[i], true_top_atoms[i], k)
            mp.append(masked_pred(g, matched_atoms))
        matched_preds_all.append(np.array(mp))

    return labels, base_preds, true_preds, perm_preds_all, rand_preds_all, matched_preds_all, task


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap CI 계산
# ══════════════════════════════════════════════════════════════════════════════
def run_bootstrap(labels, base_preds, true_preds, perm_preds_all, rand_preds_all,
                   matched_preds_all, task, n_bootstrap):
    """Bootstrap samples → ACCS_vs_perm / ACCS_vs_matched distribution."""
    np.random.seed(SEED)
    n = len(labels)
    perm_arr    = np.stack(perm_preds_all)     # [N_REPEAT, n]
    rand_arr    = np.stack(rand_preds_all)     # [N_REPEAT, n]
    matched_arr = np.stack(matched_preds_all)  # [N_REPEAT, n]

    accs_samples, accs_perm_samples, accs_vs_perm_samples = [], [], []
    accs_matched_samples, accs_vs_matched_samples = [], []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)

        perf_base = compute_metric(base_preds[idx], labels[idx], task)
        perf_true = compute_metric(true_preds[idx], labels[idx], task)
        perf_perm = np.mean([compute_metric(perm_arr[r][idx], labels[idx], task)
                             for r in range(len(perm_preds_all))])
        perf_rand = np.mean([compute_metric(rand_arr[r][idx], labels[idx], task)
                             for r in range(len(rand_preds_all))])
        perf_matched = np.mean([compute_metric(matched_arr[r][idx], labels[idx], task)
                                for r in range(len(matched_preds_all))])

        d_true    = perf_base - perf_true
        d_perm    = perf_base - perf_perm
        d_rand    = perf_base - perf_rand
        d_matched = perf_base - perf_matched

        accs           = d_true - d_rand
        accs_perm      = d_perm - d_rand
        accs_vs_perm   = accs - accs_perm
        accs_matched   = d_matched - d_rand
        accs_vs_matched = d_true - d_matched  # true가 degree-matched보다 유의하게 더 나쁜지

        accs_samples.append(accs)
        accs_perm_samples.append(accs_perm)
        accs_vs_perm_samples.append(accs_vs_perm)
        accs_matched_samples.append(accs_matched)
        accs_vs_matched_samples.append(accs_vs_matched)

    return (accs_samples, accs_perm_samples, accs_vs_perm_samples,
            accs_matched_samples, accs_vs_matched_samples)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global MASK_RATIO
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--model",      required=True, choices=["ecfp4","gcn","gat","gin"])
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed",       type=int, default=42,
                         help="scaffold split/model seed. 42=기존 결과와 동일 경로.")
    parser.add_argument("--mask-ratio", type=float, default=MASK_RATIO,
                         help="top-k 비율. 기본 0.2 (0.1/0.3 스윕 시 지정).")
    args = parser.parse_args()
    MASK_RATIO = args.mask_ratio

    if args.dataset not in DATASETS:
        print(f"Unknown dataset: {args.dataset}"); return

    task       = DATASETS[args.dataset]["task"]
    data_dir   = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    device     = torch.device("cpu")  # bootstrap은 CPU로 충분
    _, out_dir = resolve_seed_dirs(
        args.dataset, data_dir, result_dir,
        "mlp" if args.model == "ecfp4" else f"gnn_{args.model}",
        args.seed,
    )

    print(f"\n{'='*55}")
    print(f"Bootstrap CI: {args.dataset} | {args.model.upper()} | seed={args.seed} | "
          f"mask_ratio={MASK_RATIO} | n_bootstrap={args.n_bootstrap}")

    sys.path.insert(0, str(Path(__file__).parent))

    if args.model == "ecfp4":
        result = run_ecfp4_bootstrap(args.dataset, task, data_dir, result_dir, device, seed=args.seed)
    else:
        result = run_gnn_bootstrap(args.dataset, args.model, task, data_dir, result_dir, device, seed=args.seed)

    if result is None:
        return

    labels, base_preds, true_preds, perm_preds_all, rand_preds_all, matched_preds_all, task = result
    print(f"  n_test={len(labels)}")

    # 관측값 (bootstrap 전)
    perf_base    = compute_metric(base_preds, labels, task)
    perf_true    = compute_metric(true_preds, labels, task)
    perf_perm    = np.mean([compute_metric(p, labels, task) for p in perm_preds_all])
    perf_rand    = np.mean([compute_metric(r, labels, task) for r in rand_preds_all])
    perf_matched = np.mean([compute_metric(m, labels, task) for m in matched_preds_all])
    obs_accs_vs_perm    = (perf_base - perf_true) - (perf_base - perf_perm)
    obs_accs_vs_matched = (perf_base - perf_true) - (perf_base - perf_matched)

    print(f"  Observed ACCS_vs_perm    = {obs_accs_vs_perm:+.4f}")
    print(f"  Observed ACCS_vs_matched = {obs_accs_vs_matched:+.4f}")

    # Bootstrap
    print(f"  Running {args.n_bootstrap} bootstrap iterations...", flush=True)
    accs_s, accs_perm_s, accs_vs_perm_s, accs_matched_s, accs_vs_matched_s = run_bootstrap(
        labels, base_preds, true_preds, perm_preds_all, rand_preds_all, matched_preds_all,
        task, args.n_bootstrap
    )

    mean_avp, lo_avp, hi_avp, p_avp = bootstrap_ci(accs_vs_perm_s)
    mean_a,   lo_a,   hi_a,   p_a   = bootstrap_ci(accs_s)
    mean_avm, lo_avm, hi_avm, p_avm = bootstrap_ci(accs_vs_matched_s)

    def stars(p):
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    sig     = stars(p_avp)
    sig_avm = stars(p_avm)
    print(f"  ACCS_vs_perm    = {mean_avp:+.4f} [{lo_avp:+.4f}, {hi_avp:+.4f}]  p={p_avp:.4f} {sig}")
    print(f"  ACCS_vs_matched = {mean_avm:+.4f} [{lo_avm:+.4f}, {hi_avm:+.4f}]  p={p_avm:.4f} {sig_avm}")
    print(f"  ACCS            = {mean_a:+.4f} [{lo_a:+.4f}, {hi_a:+.4f}]  p={p_a:.4f}")

    # 저장
    ci_result = {
        "dataset": args.dataset, "model": args.model, "task": task, "seed": args.seed,
        "n_test": int(len(labels)), "n_bootstrap": args.n_bootstrap,
        "mask_ratio": MASK_RATIO,
        "obs_accs_vs_perm": float(obs_accs_vs_perm),
        "bootstrap_mean": float(mean_avp),
        "ci_lo": float(lo_avp), "ci_hi": float(hi_avp),
        "p_value": float(p_avp),
        "significant": sig,
        "ACCS_mean": float(mean_a), "ACCS_ci_lo": float(lo_a), "ACCS_ci_hi": float(hi_a),
        # degree-matched baseline (permuted보다 엄격한 대조군)
        "obs_accs_vs_matched": float(obs_accs_vs_matched),
        "matched_bootstrap_mean": float(mean_avm),
        "matched_ci_lo": float(lo_avm), "matched_ci_hi": float(hi_avm),
        "matched_p_value": float(p_avm), "matched_significant": sig_avm,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ratio_tag = "" if abs(MASK_RATIO - 0.2) < 1e-9 else f"_ratio{MASK_RATIO}"
    ci_filename = f"bootstrap_ci{ratio_tag}.json"
    with open(out_dir / ci_filename, "w") as f:
        json.dump(ci_result, f, indent=2)

    # Summary CSV 업데이트 (seed=42, ratio=0.2 기본 조합만 bootstrap_summary.csv에 누적 —
    # 논문 메인 테이블과 동일 경로 유지. 그 외 seed/ratio는 09_multiseed_summary.py가
    # 개별 bootstrap_ci*.json 파일들을 직접 스캔해 집계한다.)
    if args.seed == 42 and ratio_tag == "":
        summary_path = result_dir / "bootstrap_summary.csv"
        new_row = pd.DataFrame([ci_result])
        if summary_path.exists():
            old = pd.read_csv(summary_path)
            old = old[~((old["dataset"] == args.dataset) & (old["model"] == args.model))]
            pd.concat([old, new_row], ignore_index=True).to_csv(summary_path, index=False)
        else:
            new_row.to_csv(summary_path, index=False)

    print(f"  Saved: {out_dir}/{ci_filename}")


if __name__ == "__main__":
    main()
