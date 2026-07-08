"""
02_train_models.py
===================
각 데이터셋별 ECFP4-MLP + GATv2 학습.

실행:
  cd /path/to/AttnBench
  python experiments/02_train_models.py --dataset BBBP --model mlp
  python experiments/02_train_models.py --dataset BBBP --model attentivefp
  python experiments/02_train_models.py --dataset all  --model all

  # multi-seed 재학습 (01_download_data.py --seeds 42,0,1,2,3,4 로 split 먼저 생성해둘 것):
  python experiments/02_train_models.py --dataset all --model mlp --seed 0
  python experiments/02_train_models.py --dataset all --model mlp --seed 1
  ...

출력:
  results/<dataset>/<model>/best_model.pt                (seed=42, 기존 경로 그대로)
  results/<dataset>/<model>/seed_{N}/best_model.pt         (seed=N, N != 42)
  results/<dataset>/<model>/metrics.json (or seed_{N}/metrics.json)
"""

import argparse
import json
import random
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

NBITS  = 2048
RADIUS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = ["BBBP", "BACE", "HIV", "ClinTox", "ESOL", "Lipophilicity", "FreeSolv", "Tox21"]
MODELS   = ["mlp", "attentivefp"]

TASK_TYPE = {
    "BBBP": "classification", "BACE": "classification",
    "HIV": "classification",  "ClinTox": "classification",
    "ESOL": "regression",     "Lipophilicity": "regression",
    "FreeSolv": "regression", "Tox21": "multilabel",
}

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


# ─── 분자 표현 ─────────────────────────────────────────────────────────────────
def smiles_to_ecfp(smiles_list):
    fps = []
    for smi in smiles_list:
        arr = np.zeros(NBITS, dtype=np.float32)
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS)
            DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.array(fps, dtype=np.float32)


def smiles_to_graph(smiles_list):
    """PyG Data 객체 리스트로 변환."""
    try:
        from torch_geometric.data import Data
    except ImportError:
        return None

    graphs = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            graphs.append(None)
            continue

        # 원자 특성 (9차원)
        atom_feats = []
        for atom in mol.GetAtoms():
            feat = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                int(atom.IsInRing()),
                atom.GetNumRadicalElectrons(),
                atom.GetMass() / 100.0,
            ]
            atom_feats.append(feat)

        x = torch.tensor(atom_feats, dtype=torch.float)

        # 엣지
        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges += [[i, j], [j, i]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() \
                     if edges else torch.zeros((2, 0), dtype=torch.long)

        graphs.append(Data(x=x, edge_index=edge_index))
    return graphs


# ─── 모델 정의 ────────────────────────────────────────────────────────────────
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

    def get_atom_saliency(self, smiles, target_idx=0):
        """ECFP4 비트 → 원자 saliency (gradient 기반)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        arr = np.zeros(NBITS, dtype=np.float32)
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, RADIUS, nBits=NBITS, bitInfo=bit_info)
        DataStructs.ConvertToNumpyArray(fp, arr)

        x = torch.tensor(arr, dtype=torch.float32, requires_grad=True).unsqueeze(0).to(DEVICE)
        self.eval()
        out = self(x)
        if out.dim() > 0 and out.shape[0] > 1:
            out = out[target_idx]
        grad = torch.autograd.grad(out, x)[0].squeeze().abs().cpu().numpy()

        # 비트 → 원자 매핑
        n_atoms = mol.GetNumAtoms()
        atom_scores = np.zeros(n_atoms)
        for bit, envs in bit_info.items():
            if bit < NBITS:
                for center, _ in envs:
                    atom_scores[center] += float(grad[bit])

        return atom_scores


# ─── AttentiveFP (PyG 기반) ───────────────────────────────────────────────────
def build_attentivefp(out_dim):
    try:
        from torch_geometric.nn import AttentiveFP
        return AttentiveFP(
            in_channels=9, hidden_channels=128, out_channels=out_dim,
            edge_dim=1, num_layers=3, num_timesteps=2, dropout=0.2,
        )
    except ImportError:
        return None


# ─── 학습 루프 ────────────────────────────────────────────────────────────────
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_dir(dataset: str, seed: int) -> Path:
    """seed=42는 기존 data/<dataset>/ 그대로, 그 외는 split_seed{N}/ 하위 사용."""
    base = Path("data") / dataset
    if seed == 42:
        return base
    split_dir = base / f"split_seed{seed}"
    if not (split_dir / "train.csv").exists():
        print(f"  [WARN] {split_dir} 없음 — 01_download_data.py --seeds ...{seed} 먼저 실행 필요. "
              f"기본 data/{dataset}/로 폴백합니다.")
        return base
    return split_dir


def train_mlp(dataset: str, out_dir: Path, args):
    task = TASK_TYPE[dataset]
    set_all_seeds(args.seed)
    data_dir = resolve_data_dir(dataset, args.seed)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "val.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")

    # 라벨 준비
    if task == "multilabel":
        label_cols = [c for c in train_df.columns if c != "smiles"]
        y_train = train_df[label_cols].values.astype(np.float32)
        y_val   = val_df[label_cols].values.astype(np.float32)
        y_test  = test_df[label_cols].values.astype(np.float32)
        out_dim = len(label_cols)
    else:
        y_train = train_df["label"].values.astype(np.float32)
        y_val   = val_df["label"].values.astype(np.float32)
        y_test  = test_df["label"].values.astype(np.float32)
        out_dim = 1

    X_train = smiles_to_ecfp(train_df["smiles"])
    X_val   = smiles_to_ecfp(val_df["smiles"])
    X_test  = smiles_to_ecfp(test_df["smiles"])

    Xt = torch.tensor(X_train).to(DEVICE)
    Xv = torch.tensor(X_val).to(DEVICE)
    Xe = torch.tensor(X_test).to(DEVICE)
    yt = torch.tensor(y_train).to(DEVICE)
    yv = torch.tensor(y_val).to(DEVICE)
    ye = torch.tensor(y_test).to(DEVICE)

    model = ECFP_MLP(out_dim=out_dim).to(DEVICE)

    # 손실 함수
    if task in ("classification", "multilabel"):
        if task == "classification":
            pos_w = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_metric = -np.inf  # val_metric은 분류/회귀 모두 "클수록 좋음" 부호로 통일되어 있음
                                # (회귀는 -RMSE). 예전 코드에서 회귀만 +inf로 초기화되어 있어
                                # val_metric(음수)이 영원히 이를 못 넘어서 best_state가 None으로
                                # 남는 버그가 있었음 → 항상 -inf로 통일.
    best_state = None
    patience_cnt = 0
    PATIENCE = 25

    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            xb = Xt[idx[i:i+256]]
            yb = yt[idx[i:i+256]]
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(Xv)
            if task == "regression":
                val_metric = -mean_squared_error(yv.cpu(), val_out.cpu())
            else:
                probs = torch.sigmoid(val_out).cpu().numpy()
                try:
                    if task == "multilabel":
                        # 유효한 컬럼만 AUROC 계산
                        aucs = []
                        for j in range(probs.shape[1]):
                            mask = ~np.isnan(y_val[:, j])
                            if mask.sum() > 0 and len(np.unique(y_val[mask, j])) > 1:
                                aucs.append(roc_auc_score(y_val[mask, j], probs[mask, j]))
                        val_metric = np.mean(aucs) if aucs else 0.5
                    else:
                        val_metric = roc_auc_score(y_val, probs.squeeze())
                except Exception:
                    val_metric = 0.5

        scheduler.step(-val_metric if task != "regression" else -val_metric)

        improved = val_metric > best_val_metric
        if improved:
            best_val_metric = val_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}: val={val_metric:.4f} (best={best_val_metric:.4f})")

    # 최종 평가
    if best_state is None:
        raise RuntimeError(
            f"[{dataset}] best_state가 None입니다 — 첫 epoch부터 개선이 감지되지 않았습니다. "
            f"학습률/데이터 문제일 수 있으니 확인하세요 (val_metric 로그 참고).")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_out = model(Xe)

    metrics = {"dataset": dataset, "model": "mlp", "task": task, "seed": args.seed}
    if task == "regression":
        preds = test_out.cpu().numpy()
        metrics["rmse"]  = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics["mae"]   = float(mean_absolute_error(y_test, preds))
        metrics["r2"]    = float(r2_score(y_test, preds))
        print(f"  Test RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
    else:
        probs = torch.sigmoid(test_out).cpu().numpy()
        if task == "multilabel":
            aucs = []
            for j in range(probs.shape[1]):
                mask = ~np.isnan(y_test[:, j])
                if mask.sum() > 0 and len(np.unique(y_test[mask, j])) > 1:
                    aucs.append(roc_auc_score(y_test[mask, j], probs[mask, j]))
            metrics["auroc_mean"] = float(np.mean(aucs))
            print(f"  Test AUROC (mean)={metrics['auroc_mean']:.4f}")
        else:
            metrics["auroc"] = float(roc_auc_score(y_test, probs.squeeze()))
            print(f"  Test AUROC={metrics['auroc']:.4f}")

    torch.save({"state_dict": best_state, "out_dim": out_dim,
                "task": task, "metrics": metrics}, out_dir / "best_model.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # test predictions 저장 (saliency 분석용)
    test_df_out = test_df.copy()
    if task == "multilabel":
        label_cols_avail = [c for c in test_df.columns if c != "smiles"]
        probs_df = pd.DataFrame(
            torch.sigmoid(test_out).cpu().numpy(),
            columns=[f"pred_{c}" for c in label_cols_avail])
        test_df_out = pd.concat([test_df_out.reset_index(drop=True), probs_df], axis=1)
    else:
        test_df_out["pred"] = torch.sigmoid(test_out).cpu().numpy() \
                              if task == "classification" \
                              else test_out.cpu().numpy()
    test_df_out.to_csv(out_dir / "test_predictions.csv", index=False)
    return metrics


# ─── 메인 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--model",   default="mlp", choices=["mlp", "attentivefp", "all"])
    parser.add_argument("--epochs",  type=int, default=150)
    parser.add_argument("--seed",    type=int, default=42,
                         help="scaffold split seed + model init seed. 42=기존 결과와 동일 경로.")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    models   = ["mlp"] if args.model == "attentivefp" else \
               MODELS if args.model == "all" else [args.model]

    all_metrics = []
    for ds in datasets:
        data_dir = resolve_data_dir(ds, args.seed)
        if not (data_dir / "train.csv").exists():
            print(f"[SKIP] {ds}: data not found. Run 01_download_data.py first.")
            continue
        for model in models:
            if model != "mlp":
                print(f"[SKIP] {model}: not yet implemented — use mlp for now")
                continue
            out_dir = Path("results") / ds / model
            if args.seed != 42:
                out_dir = out_dir / f"seed_{args.seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*55}\nDataset: {ds}  Model: {model}  Seed: {args.seed}\n{'='*55}")
            m = train_mlp(ds, out_dir, args)
            all_metrics.append(m)

    if all_metrics:
        summary = pd.DataFrame(all_metrics)
        Path("results").mkdir(exist_ok=True)
        summary_name = "results/training_summary.csv" if args.seed == 42 \
                       else f"results/training_summary_seed{args.seed}.csv"
        summary.to_csv(summary_name, index=False)
        print(f"\n{'='*55}")
        print("Training complete!")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
