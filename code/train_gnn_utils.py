"""
train_gnn_utils.py
===================
04_train_gnn.py 와 05_accs_gnn.py 공유 코드.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from rdkit import Chem

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool

# ─── 데이터셋 메타 ─────────────────────────────────────────────────────────────
DATASETS = {
    "BBBP":         {"task": "classification", "label_col": "label"},
    "BACE":         {"task": "classification", "label_col": "label"},
    "HIV":          {"task": "classification", "label_col": "label"},
    "ClinTox":      {"task": "classification", "label_col": "label"},
    "ESOL":         {"task": "regression",     "label_col": "label"},
    "Lipophilicity":{"task": "regression",     "label_col": "label"},
    "FreeSolv":     {"task": "regression",     "label_col": "label"},
}

# ─── 원자 특징 ─────────────────────────────────────────────────────────────────
ATOM_TYPES = [
    'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe',
    'As','Al','I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co',
    'Se','Ti','Zn','H','Li','Ge','Cu','Au','Ni','Cd','In','Mn',
    'Zr','Cr','Pt','Hg','Pb','other'
]

HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

ATOM_DIM = len(ATOM_TYPES) + 1 + 1 + 1 + len(HYBRIDIZATION) + 1 + 1


def one_hot(val, choices):
    vec = [0] * len(choices)
    if val in choices:
        vec[choices.index(val)] = 1
    else:
        vec[-1] = 1
    return vec


def atom_features(atom):
    feats = one_hot(atom.GetSymbol(), ATOM_TYPES)
    feats += [atom.GetDegree() / 10.0]
    feats += [atom.GetFormalCharge() / 5.0]
    feats += [int(atom.GetIsAromatic())]
    feats += one_hot(atom.GetHybridization(), HYBRIDIZATION)
    feats += [atom.GetTotalNumHs() / 4.0]
    feats += [int(atom.IsInRing())]
    return feats


def smiles_to_data(smiles: str, label, task: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    xs = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(xs, dtype=torch.float)

    rows, cols = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [i, j]; cols += [j, i]

    if rows:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    if task == "classification":
        y = torch.tensor([int(label)], dtype=torch.long)
    else:
        y = torch.tensor([float(label)], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)


def load_dataset(data_dir: Path, task: str, label_col: str):
    splits = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(data_dir / f"{split}.csv")
        graphs = []
        for _, row in df.iterrows():
            g = smiles_to_data(row["smiles"], row[label_col], task)
            if g is not None:
                graphs.append(g)
        splits[split] = graphs
    return splits


# ─── 모델 ─────────────────────────────────────────────────────────────────────
class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=1, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.pool  = global_mean_pool
        self.head  = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.pool(x, batch)
        return self.head(x).squeeze(-1)


class GINModel(nn.Module):
    """Graph Isomorphism Network (Xu et al., 2019).

    GCN/GAT과 동일한 forward 시그니처 (x, edge_index, batch)를 유지하여
    05_accs_gnn / 06_bootstrap_ci / 08_xai_baselines 에서 GCN과 동일한
    gradient saliency 경로로 재사용된다. GIN은 attention을 쓰지 않으므로
    native attribution은 node-feature gradient (GCN과 동일)이다.
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=1, dropout=0.2):
        super().__init__()

        def mlp(i, o):
            return nn.Sequential(
                nn.Linear(i, o), nn.ReLU(),
                nn.Linear(o, o),
            )

        # train_eps=True: self-node 가중치 학습 (표준 GIN-eps)
        self.conv1 = GINConv(mlp(in_dim,     hidden_dim), train_eps=True)
        self.conv2 = GINConv(mlp(hidden_dim, hidden_dim), train_eps=True)
        self.conv3 = GINConv(mlp(hidden_dim, hidden_dim), train_eps=True)
        self.pool  = global_mean_pool
        self.head  = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.pool(x, batch)
        return self.head(x).squeeze(-1)


class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1, heads=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(in_dim,          hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim*heads, hidden_dim, heads=1,     dropout=dropout, concat=False)
        self.pool  = global_mean_pool
        self.head  = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x, edge_index, batch, return_attn=False):
        x, a1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x, a2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x, a3 = self.conv3(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.pool(x, batch)
        out = self.head(x).squeeze(-1)
        if return_attn:
            return out, (a1, a2, a3)
        return out
