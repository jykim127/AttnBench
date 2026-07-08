"""
04_train_gnn.py
================
GCN + GAT 모델 학습 (MoleculeNet 7 datasets).
ECFP4-MLP과 동일한 scaffold split 사용.

설치 (처음 한 번):
  pip install torch-geometric --break-system-packages
  # CUDA 버전에 맞는 scatter/sparse 필요 시:
  # pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-<ver>.html

실행:
  cd /path/to/AttnBench
  python experiments/04_train_gnn.py --dataset BACE

  # multi-seed 재학습 (01_download_data.py --seeds 42,0,1,2,3,4 로 split 먼저 생성해둘 것):
  python experiments/04_train_gnn.py --dataset BACE --seed 0

  # 전체 백그라운드:
  nohup python -c "
  import subprocess, sys
  datasets = ['BBBP','BACE','HIV','ClinTox','ESOL','Lipophilicity','FreeSolv']
  for ds in datasets:
      print(f'=== {ds} ===', flush=True)
      subprocess.run([sys.executable, 'experiments/04_train_gnn.py', '--dataset', ds])
  " > logs/gnn_train.log 2>&1 &

출력:
  results/<dataset>/gnn_gcn/best_model.pt + metrics.json
  results/<dataset>/gnn_gat/best_model.pt + metrics.json
"""

import argparse
import json
import random
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, mean_squared_error

try:
    from torch_geometric.data import DataLoader
except ImportError:
    print("[ERROR] pip install torch-geometric --break-system-packages")
    raise

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_gnn_utils import DATASETS, load_dataset, GCNModel, GATModel, GINModel


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_dir(dataset: str, data_root: Path, seed: int) -> Path:
    """seed=42는 기존 data/<dataset>/ 그대로, 그 외는 split_seed{N}/ 하위 사용."""
    base = data_root / dataset
    if seed == 42:
        return base
    split_dir = base / f"split_seed{seed}"
    if not (split_dir / "train.csv").exists():
        print(f"  [WARN] {split_dir} 없음 — 01_download_data.py --seeds ...{seed} 먼저 실행 필요. "
              f"기본 {base}/로 폴백합니다.")
        return base
    return split_dir


# ─── 평가 ──────────────────────────────────────────────────────────────────────
def evaluate(model, loader, task, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds.append(out.cpu().numpy())
            labels.append(batch.y.cpu().numpy())
    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)

    if task == "classification":
        probs = 1 / (1 + np.exp(-preds))
        try:
            return roc_auc_score(labels, probs)
        except Exception:
            return 0.5
    else:
        return -np.sqrt(mean_squared_error(labels, preds))


# ─── 학습 ──────────────────────────────────────────────────────────────────────
def train_model(model, splits, task, device, lr=1e-3, epochs=150, patience=20, batch_size=64):
    train_loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(splits["val"],   batch_size=batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    best_val, best_state, no_improve = -np.inf, None, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            out  = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.float())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        vm = evaluate(model, val_loader, task, device)
        scheduler.step(-vm)

        if vm > best_val:
            best_val   = vm
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={total_loss/len(train_loader):.4f} val={vm:.4f}")

    model.load_state_dict(best_state)
    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir",  default="results")
    parser.add_argument("--epochs",   type=int, default=150)
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--seed",     type=int, default=42,
                         help="scaffold split seed + model init seed. 42=기존 결과와 동일 경로.")
    parser.add_argument("--models",   default="gcn,gat",
                         help="학습할 모델 (콤마 구분). 예: gin  또는  gcn,gat,gin. "
                              "기존 GCN/GAT 재학습 없이 GIN만 추가하려면 --models gin.")
    args = parser.parse_args()

    model_types = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    valid = {"gcn", "gat", "gin"}
    bad = [m for m in model_types if m not in valid]
    if bad:
        print(f"[ERROR] Unknown model(s): {bad}. Choose from {sorted(valid)}."); return

    if args.dataset not in DATASETS:
        print(f"Unknown dataset: {args.dataset}"); return

    set_all_seeds(args.seed)
    meta   = DATASETS[args.dataset]
    task   = meta["task"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"Dataset: {args.dataset} | Task: {task} | Device: {device} | Seed: {args.seed}")

    data_dir = resolve_data_dir(args.dataset, Path(args.data_dir), args.seed)
    splits = load_dataset(data_dir, task, meta["label_col"])
    print(f"  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    if len(splits["train"]) == 0:
        print("[ERROR] No training data."); return

    in_dim = splits["train"][0].x.shape[1]
    test_loader = DataLoader(splits["test"], batch_size=64, shuffle=False)

    for model_type in model_types:
        print(f"\n--- {model_type.upper()} ---")
        out_dir = Path(args.out_dir) / args.dataset / f"gnn_{model_type}"
        if args.seed != 42:
            out_dir = out_dir / f"seed_{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if model_type == "gcn":
            model = GCNModel(in_dim)
        elif model_type == "gin":
            model = GINModel(in_dim)
        else:
            model = GATModel(in_dim, hidden_dim=64, heads=4)
        model = model.to(device)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        best_val    = train_model(model, splits, task, device, epochs=args.epochs, batch_size=args.batch)
        test_metric = evaluate(model, test_loader, task, device)
        metric_name = "AUROC" if task == "classification" else "neg_RMSE"
        print(f"  Best val {metric_name}: {best_val:.4f}  Test: {test_metric:.4f}")

        torch.save(model.state_dict(), out_dir / "best_model.pt")
        with open(out_dir / "metrics.json", "w") as f:
            json.dump({
                "dataset": args.dataset, "task": task, "model": model_type,
                "val_metric": float(best_val), "test_metric": float(test_metric),
                "metric_name": metric_name, "seed": args.seed,
            }, f, indent=2)
        print(f"  Saved → {out_dir}/")


if __name__ == "__main__":
    main()
