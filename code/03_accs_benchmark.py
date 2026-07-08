"""
03_accs_benchmark.py
=====================
핵심 실험: ACCS (Attention Causal Contribution Score) 계산.

방법:
  1. 학습된 모델에서 gradient saliency로 원자별 중요도 점수 계산
  2. 세 가지 원자 마스킹 조건 비교:
     (A) True saliency: 상위 k 원자 마스킹
     (B) Permuted saliency: 점수를 원자 간 랜덤 셔플 후 상위 k 마스킹
     (C) Random: 완전 무작위 k 원자 마스킹
  3. 각 조건에서 예측 성능 측정
  4. ACCS = (Δperf_true - Δperf_random) / baseline_perf

실행:
  cd /path/to/AttnBench
  python experiments/03_accs_benchmark.py --dataset BBBP
  python experiments/03_accs_benchmark.py --dataset all

출력:
  results/<dataset>/mlp/accs_results.json
  results/accs_summary.csv
  results/fig_accs_benchmark.png
"""

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy import stats

NBITS  = 2048
RADIUS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_RATIOS = [0.1, 0.2, 0.3]   # top-k% 원자 마스킹
N_PERMUTE = 20                   # 랜덤 셔플 반복 횟수

TASK_TYPE = {
    "BBBP": "classification", "BACE": "classification",
    "HIV": "classification",  "ClinTox": "classification",
    "ESOL": "regression",     "Lipophilicity": "regression",
    "FreeSolv": "regression", "Tox21": "multilabel",
}


# ─── 모델 로드 ────────────────────────────────────────────────────────────────
class ECFP_MLP(nn.Module):
    def __init__(self, out_dim=1, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NBITS, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),   nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model(model_dir: Path):
    ckpt = torch.load(model_dir / "best_model.pt", map_location=DEVICE)
    out_dim = ckpt.get("out_dim", 1)
    model = ECFP_MLP(out_dim=out_dim).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["task"]


# ─── ECFP4 비트 → 원자 saliency ──────────────────────────────────────────────
def get_atom_saliency(model, smiles, task, target_idx=0):
    """
    단일 분자에 대한 원자별 gradient saliency.
    반환: np.array(n_atoms,)
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return None, None

    arr = np.zeros(NBITS, dtype=np.float32)
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS, bitInfo=bit_info)
    DataStructs.ConvertToNumpyArray(fp, arr)

    x = torch.tensor(arr, dtype=torch.float32, requires_grad=True).unsqueeze(0).to(DEVICE)
    out = model(x)

    # scalar 추출
    if out.dim() == 0:
        scalar = out
    elif out.dim() == 1 and out.shape[0] == 1:
        scalar = out[0]
    elif out.dim() == 1:
        scalar = out[target_idx]
    else:
        scalar = out[0, target_idx]

    grad = torch.autograd.grad(scalar, x)[0].squeeze().abs().detach().cpu().numpy()

    # 비트 → 원자 saliency 집계
    n_atoms = mol.GetNumAtoms()
    atom_scores = np.zeros(n_atoms)
    for bit, envs in bit_info.items():
        if bit < NBITS:
            for center, _ in envs:
                if center < n_atoms:
                    atom_scores[center] += float(grad[bit])

    return atom_scores, bit_info


# ─── 원자 마스킹 ─────────────────────────────────────────────────────────────
def mask_atoms_and_predict(model, smiles, atom_scores, mask_ratio, task,
                           permuted=False, random_mask=False, rng=None):
    """
    상위 mask_ratio 원자를 마스킹 후 예측값 반환.
    마스킹: 해당 원자가 속한 ECFP4 비트를 0으로 설정.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    # 마스킹할 원자 결정
    k = max(1, int(n_atoms * mask_ratio))
    scores = atom_scores.copy()

    if permuted:
        rng_ = rng if rng else np.random.default_rng()
        scores = rng_.permutation(scores)
    elif random_mask:
        rng_ = rng if rng else np.random.default_rng()
        scores = rng_.random(n_atoms)

    mask_atoms = set(np.argsort(scores)[-k:])

    # 마스킹된 fingerprint 계산
    arr = np.zeros(NBITS, dtype=np.float32)
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS, bitInfo=bit_info)
    DataStructs.ConvertToNumpyArray(fp, arr)

    # 마스킹된 원자에서 비롯된 비트 → 0으로
    arr_masked = arr.copy()
    for bit, envs in bit_info.items():
        if bit < NBITS:
            for center, _ in envs:
                if center in mask_atoms:
                    arr_masked[bit] = 0.0
                    break

    x = torch.tensor(arr_masked, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)

    if out.dim() == 0:
        return out.item()
    elif out.dim() == 1 and out.shape[0] == 1:
        return out[0].item()
    elif task in ("classification", "regression"):
        return out[0].item()
    else:
        return out[0].cpu().numpy()


# ─── ACCS 계산 ───────────────────────────────────────────────────────────────
def compute_accs(dataset: str, model_dir: Path, out_dir: Path, args):
    model, task = load_model(model_dir)
    test_df = pd.read_csv(Path("data") / dataset / "test.csv")
    smiles_list = test_df["smiles"].tolist()

    if task == "multilabel":
        label_cols = [c for c in test_df.columns if c != "smiles"]
        y_true = test_df[label_cols].values.astype(np.float32)
    else:
        y_true = test_df["label"].values.astype(np.float32)

    print(f"  Computing saliency for {len(smiles_list)} molecules...")
    rng = np.random.default_rng(42)

    # baseline 예측 (마스킹 없음)
    preds_baseline = []
    all_atom_scores = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            preds_baseline.append(None)
            all_atom_scores.append(None)
            continue

        arr = np.zeros(NBITS, dtype=np.float32)
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS, bitInfo=bit_info)
        DataStructs.ConvertToNumpyArray(fp, arr)

        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)

        if task == "multilabel":
            preds_baseline.append(torch.sigmoid(out[0]).cpu().numpy())
        elif task == "classification":
            preds_baseline.append(torch.sigmoid(out).item())
        else:
            preds_baseline.append(out.item())

        # saliency
        scores, _ = get_atom_saliency(model, smi, task)
        all_atom_scores.append(scores)

    results_per_ratio = []

    for mask_ratio in MASK_RATIOS:
        print(f"  Mask ratio: {mask_ratio*100:.0f}%")

        preds_true = []
        preds_perm_all = [[] for _ in range(N_PERMUTE)]
        preds_rand_all = [[] for _ in range(N_PERMUTE)]

        for idx, (smi, scores) in enumerate(zip(smiles_list, all_atom_scores)):
            if scores is None:
                preds_true.append(None)
                for j in range(N_PERMUTE):
                    preds_perm_all[j].append(None)
                    preds_rand_all[j].append(None)
                continue

            # True saliency 마스킹
            pt = mask_atoms_and_predict(model, smi, scores, mask_ratio, task)
            preds_true.append(pt)

            # Permuted 마스킹 (N_PERMUTE회)
            for j in range(N_PERMUTE):
                pp = mask_atoms_and_predict(model, smi, scores, mask_ratio, task,
                                            permuted=True, rng=rng)
                preds_perm_all[j].append(pp)

                pr = mask_atoms_and_predict(model, smi, scores, mask_ratio, task,
                                            random_mask=True, rng=rng)
                preds_rand_all[j].append(pr)

        # 유효한 인덱스
        valid = [i for i, (b, t) in enumerate(zip(preds_baseline, preds_true))
                 if b is not None and t is not None]
        if not valid:
            continue

        def eval_perf(preds, indices):
            """성능 지표 계산."""
            if task == "regression":
                y = y_true[indices]
                p = np.array([preds[i] for i in indices])
                return -np.sqrt(mean_squared_error(y, p))   # -RMSE (높을수록 좋음)
            elif task == "classification":
                y = y_true[indices]
                p = torch.sigmoid(torch.tensor([preds[i] for i in indices])).numpy()
                try:
                    return roc_auc_score(y, p)
                except Exception:
                    return 0.5
            else:  # multilabel
                aucs = []
                pmat = np.array([torch.sigmoid(torch.tensor(preds[i])).numpy()
                                 if not isinstance(preds[i], np.ndarray)
                                 else preds[i] for i in indices])
                for j in range(y_true.shape[1]):
                    mask = ~np.isnan(y_true[indices, j])
                    if mask.sum() > 0 and len(np.unique(y_true[indices[mask[0]], j:j+1])) > 1:
                        try:
                            aucs.append(roc_auc_score(y_true[indices, j][mask], pmat[:, j][mask]))
                        except Exception:
                            pass
                return np.mean(aucs) if aucs else 0.5

        perf_baseline = eval_perf(preds_baseline, valid)
        perf_true     = eval_perf(preds_true, valid)

        perms_perf = [eval_perf(preds_perm_all[j], valid) for j in range(N_PERMUTE)]
        rands_perf = [eval_perf(preds_rand_all[j], valid) for j in range(N_PERMUTE)]

        perf_perm = np.mean(perms_perf)
        perf_rand = np.mean(rands_perf)

        # ACCS
        delta_true = perf_baseline - perf_true   # 마스킹으로 인한 성능 하락 (양수 = 중요)
        delta_perm = perf_baseline - perf_perm
        delta_rand = perf_baseline - perf_rand

        # ACCS = (true 마스킹이 랜덤 마스킹보다 얼마나 더 성능을 떨어뜨리나)
        # ACCS > 0 → attention이 실제로 중요한 원자를 찾음
        accs = delta_true - delta_rand
        accs_vs_perm = delta_true - delta_perm

        # Wilcoxon test: true masking vs random masking paired
        _, p_vs_rand = stats.wilcoxon(
            [eval_perf(preds_true, valid)],
            [eval_perf(preds_rand_all[j], valid) for j in range(N_PERMUTE)][:1],
            zero_method="wilcox", alternative="less",
        ) if N_PERMUTE > 1 else (np.nan, np.nan)

        result = {
            "dataset":         dataset,
            "mask_ratio":      mask_ratio,
            "n_valid":         len(valid),
            "perf_baseline":   float(perf_baseline),
            "perf_true":       float(perf_true),
            "perf_perm_mean":  float(perf_perm),
            "perf_rand_mean":  float(perf_rand),
            "delta_true":      float(delta_true),
            "delta_perm":      float(delta_perm),
            "delta_rand":      float(delta_rand),
            "ACCS":            float(accs),            # primary metric
            "ACCS_vs_perm":    float(accs_vs_perm),
        }
        results_per_ratio.append(result)

        print(f"    baseline={perf_baseline:.4f}  true={perf_true:.4f}  "
              f"perm={perf_perm:.4f}  rand={perf_rand:.4f}")
        print(f"    ΔACCS (true-rand): {accs:+.4f}  ΔACCS (true-perm): {accs_vs_perm:+.4f}")

    # 저장
    with open(out_dir / "accs_results.json", "w") as f:
        json.dump(results_per_ratio, f, indent=2)

    # 대표 mask_ratio 0.2 결과 반환
    mid = [r for r in results_per_ratio if r["mask_ratio"] == 0.2]
    return mid[0] if mid else (results_per_ratio[0] if results_per_ratio else None)


# ─── 메인 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    datasets_all = ["BBBP", "BACE", "HIV", "ClinTox",
                    "ESOL", "Lipophilicity", "FreeSolv", "Tox21"]
    datasets = datasets_all if args.dataset == "all" else [args.dataset]

    summary = []
    for ds in datasets:
        model_dir = Path("results") / ds / "mlp"
        if not (model_dir / "best_model.pt").exists():
            print(f"[SKIP] {ds}: no trained model. Run 02_train_models.py first.")
            continue
        print(f"\n{'='*55}\n[ACCS] Dataset: {ds}\n{'='*55}")
        result = compute_accs(ds, model_dir, model_dir, args)
        if result:
            summary.append(result)

    if not summary:
        print("No results.")
        return

    df = pd.DataFrame(summary)
    df.to_csv("results/accs_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("ACCS Benchmark Summary (mask_ratio=0.2):")
    print(f"{'Dataset':<18} {'Task':<16} {'Baseline':>9} {'ACCS':>8} {'ACCS_perm':>10}")
    print("-" * 65)
    for _, row in df.iterrows():
        task = TASK_TYPE.get(row["dataset"], "?")
        print(f"{row['dataset']:<18} {task:<16} {row['perf_baseline']:>9.4f} "
              f"{row['ACCS']:>+8.4f} {row['ACCS_vs_perm']:>+10.4f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)

        # Panel A: ACCS per dataset (bar)
        ax = axes[0]
        colors = ["#0072B2" if TASK_TYPE.get(d,"") == "classification"
                  else "#E69F00" if TASK_TYPE.get(d,"") == "regression"
                  else "#009E73"
                  for d in df["dataset"]]
        bars = ax.bar(df["dataset"], df["ACCS"], color=colors, edgecolor="white", width=0.6)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_ylabel("ACCS (True − Random masking Δperf)", fontsize=10)
        ax.set_title("Attention Causal Contribution Score\nper Dataset (mask=20%)",
                     fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#0072B2", label="Classification"),
            Patch(facecolor="#E69F00", label="Regression"),
            Patch(facecolor="#009E73", label="Multi-label"),
        ]
        ax.legend(handles=legend_elements, fontsize=8)

        # Panel B: delta true / perm / rand grouped bar
        ax = axes[1]
        x = np.arange(len(df))
        w = 0.25
        ax.bar(x - w, df["delta_true"], width=w, label="True saliency", color="#0072B2", alpha=0.85)
        ax.bar(x,     df["delta_perm"], width=w, label="Permuted saliency", color="#CC79A7", alpha=0.85)
        ax.bar(x + w, df["delta_rand"], width=w, label="Random masking", color="#999999", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df["dataset"], rotation=35, fontsize=8)
        ax.set_ylabel("Δ Performance after masking", fontsize=10)
        ax.set_title("Performance Drop by Masking Strategy\n(larger = more important atoms masked)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        fig.savefig("results/fig_accs_benchmark.png", dpi=200,
                    bbox_inches="tight", facecolor="white")
        plt.close()
        print("\nSaved: results/fig_accs_benchmark.png")
    except Exception as e:
        print(f"[WARN] Figure failed: {e}")


if __name__ == "__main__":
    main()
