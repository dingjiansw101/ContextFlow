#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud


# -----------------------------
# Utils
# -----------------------------

EP_PAT = re.compile(r"episode_(\d+)\.parquet$", re.IGNORECASE)

def parse_episode_idx(name: str) -> Optional[int]:
    m = EP_PAT.search(os.path.basename(name.strip()))
    return int(m.group(1)) if m else None

def load_states_cache(path: str) -> Dict[int, np.ndarray]:
    obj = json.loads(Path(path).read_text())
    out: Dict[int, np.ndarray] = {}
    for k, v in obj.items():
        try:
            ep = int(k)
        except Exception:
            continue
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Episode {k}: expected 2D state array, got {arr.shape}")
        out[ep] = arr
    if not out:
        raise ValueError("Empty states cache after parsing")
    return out

def load_segments_json(path: str) -> List[dict]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("segments_json must be a JSON list")
    return data

def load_episodes_filter(path: Optional[str]) -> Optional[set[int]]:
    if not path:
        return None
    names = json.loads(Path(path).read_text())
    if not isinstance(names, list):
        raise ValueError("episodes_json must be a JSON list of episode filenames")
    idxs = set()
    for s in names:
        if not isinstance(s, str): 
            continue
        ep = parse_episode_idx(s)
        if ep is not None:
            idxs.add(ep)
    if not idxs:
        raise ValueError("episodes_json parsed but no valid episode indices found")
    return idxs

def build_segments_map(seg_list: List[dict]) -> Dict[int, List[dict]]:
    """Return {ep_idx: [ {id,label,start,end}, ... ]}"""
    out: Dict[int, List[dict]] = {}
    for entry in seg_list:
        name = entry.get("episode_name") or entry.get("episode_path") or ""
        ep = parse_episode_idx(str(name))
        if ep is None:
            continue
        segs = entry.get("segments", [])
        good = []
        for s in segs:
            if not isinstance(s, dict): 
                continue
            st = int(s.get("start", 0))
            ed = int(s.get("end", st))
            if ed > st:
                good.append({
                    "id": int(s.get("id", len(good))),
                    "label": str(s.get("label", "")),
                    "start": st,
                    "end": ed,
                })
        if good:
            out[ep] = good
    return out

def load_norm_stats(stats_path: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not stats_path:
        return None
    obj = json.loads(Path(stats_path).read_text())
    ns = obj.get("norm_stats", obj).get("state", None)
    if ns is None:
        raise ValueError("norm stats JSON missing 'state'")
    return np.asarray(ns["mean"], np.float32), np.asarray(ns["std"], np.float32)

def maybe_norm(x: np.ndarray, norm: Optional[Tuple[np.ndarray,np.ndarray]]) -> np.ndarray:
    if norm is None:
        return x
    mean, std = norm
    if x.shape[-1] != mean.shape[0]:
        raise ValueError(f"Norm dim mismatch: state_dim={x.shape[-1]} vs stats={mean.shape[0]}")
    return (x - mean) / (std + 1e-6)


# -----------------------------
# Dataset build
# -----------------------------

def remap_labels_and_build_meta(
    seg_map: Dict[int, List[dict]],
    use_label_name: bool
) -> Tuple[Dict[int, List[dict]], Dict[str,int], List[str]]:
    """
    Build global contiguous class ids.
    Returns: (seg_map_remapped, key2id, id2name)
    key = label string if use_label_name else original int id (as string).
    """
    keys = []
    for ep, segs in seg_map.items():
        for s in segs:
            key = s["label"] if use_label_name else str(int(s["id"]))
            keys.append(key)
    uniq = sorted(set(keys))
    key2id = {k:i for i,k in enumerate(uniq)}
    id2name = uniq[:]  # for pretty print

    # remap each segment to new id
    seg_map2: Dict[int,List[dict]] = {}
    for ep, segs in seg_map.items():
        L = []
        for s in segs:
            key = s["label"] if use_label_name else str(int(s["id"]))
            L.append({**s, "cls": key2id[key]})
        seg_map2[ep] = L
    return seg_map2, key2id, id2name

def build_samples(
    states_map: Dict[int, np.ndarray],
    seg_map: Dict[int, List[dict]],
    history: int,
    episodes_filter: Optional[set[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return X:[N, H*D], y:[N], using segment labels per frame.
    Frames outside any segment are skipped.
    """
    H = max(1, int(history))
    Xs, ys = [], []
    used_eps = 0

    for ep, states in states_map.items():
        if episodes_filter is not None and ep not in episodes_filter:
            continue
        segs = seg_map.get(ep)
        if not segs:
            continue

        T, D = states.shape
        # build per-frame label with -1 default
        lab = np.full((T,), -1, dtype=np.int32)
        for s in segs:
            st = max(0, int(s["start"]))
            ed = min(T, int(s["end"]))
            if ed > st:
                lab[st:ed] = int(s["cls"])

        valid_idx = np.nonzero(lab >= 0)[0]
        if valid_idx.size == 0:
            continue

        used_eps += 1
        for t in valid_idx:
            s = max(0, t - H + 1)
            window = states[s:t+1]  # [L,D]
            if window.shape[0] < H:
                pad = np.repeat(window[:1], H - window.shape[0], axis=0)
                window = np.concatenate([pad, window], axis=0)
            Xs.append(window.reshape(-1))
            ys.append(lab[t])

    if not Xs:
        raise ValueError("No samples built. Check filters/segments coverage.")
    X = np.asarray(Xs, np.float32)
    y = np.asarray(ys, np.int64)
    return X, y


# -----------------------------
# Model
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, dropout: float=0.0):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Train
# -----------------------------

def train(args):
    states_map = load_states_cache(args.states_cache)  # 已归一化
    seg_list = load_segments_json(args.segments_json)
    seg_map_raw = build_segments_map(seg_list)
    eps_filter = load_episodes_filter(args.episodes_json)

    # 过滤 seg_map 到白名单范围（若提供）
    if eps_filter is not None:
        seg_map_raw = {ep: segs for ep, segs in seg_map_raw.items() if ep in eps_filter}

    # 重映射类标
    seg_map, key2id, id2name = remap_labels_and_build_meta(seg_map_raw, args.use_label_name)
    num_classes = len(id2name)
    if num_classes < 2:
        raise ValueError(f"Need >=2 classes, got {num_classes}")

    # 构建样本
    X, y = build_samples(states_map, seg_map, args.history, eps_filter)
    N, in_dim = X.shape
    print(f"[data] N={N}, in_dim={in_dim}, num_classes={num_classes}, history={args.history}")

    # train/val split （按样本随机）
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(N)
    n_val = max(2000, int(0.05 * N))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xtr, ytr = torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx])
    Xva, yva = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])

    tr_loader = tud.DataLoader(tud.TensorDataset(Xtr, ytr), batch_size=args.bs, shuffle=True, drop_last=True)
    va_loader = tud.DataLoader(tud.TensorDataset(Xva, yva), batch_size=args.bs, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MLP(in_dim, [int(h) for h in args.hidden.split(",")], num_classes, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 可选类别不平衡权重
    if args.class_weight:
        counts = np.bincount(y, minlength=num_classes).astype(np.float32)
        w = (1.0 / np.maximum(counts, 1.0))
        w = w * (num_classes / np.sum(w))  # 归一化到 ~1
        loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(w).to(device))
        print(f"[info] class weights = {w.tolist()}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    best = 1e9
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = Path(args.out_dir) / "stage_head.pt"
    meta = {
        "in_dim": in_dim,
        "history": int(args.history),
        "hidden": args.hidden,
        "dropout": float(args.dropout),
        "num_classes": int(num_classes),
        "use_label_name": bool(args.use_label_name),
        "id2name": id2name,     # index -> human name（若 use_label_name=False，这里是原 id 的字符串）
        "states_cache": str(Path(args.states_cache).resolve()),
    }

    for e in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_acc, n_tr = 0.0, 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += loss.item() * xb.size(0)
            tr_acc  += (logits.argmax(1) == yb).sum().item()
            n_tr    += xb.size(0)

        tr_loss /= max(1, n_tr); tr_acc /= max(1, n_tr)

        model.eval()
        va_loss, va_acc, n_va = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                va_loss += loss.item() * xb.size(0)
                va_acc  += (logits.argmax(1) == yb).sum().item()
                n_va    += xb.size(0)
        va_loss /= max(1, n_va); va_acc /= max(1, n_va)

        print(f"[epoch {e:03d}] train_loss={tr_loss:.6f} acc={tr_acc:.4f} | val_loss={va_loss:.6f} acc={va_acc:.4f}")

        if va_loss + 1e-6 < best:
            best = va_loss
            torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt)
            (Path(args.out_dir) / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"  ↳ saved best to {ckpt}")

    print(f"done. best val loss={best:.6f}")


# -----------------------------
# Inference (online)
# -----------------------------

class StagePredictor:
    def __init__(self, ckpt_dir: str, norm_stats_json: Optional[str] = None):
        payload = torch.load(Path(ckpt_dir) / "stage_head.pt", map_location="cpu")
        self.meta = payload["meta"]
        self.model = MLP(
            in_dim=self.meta["in_dim"],
            hidden=[int(h) for h in str(self.meta["hidden"]).split(",")],
            out_dim=int(self.meta["num_classes"]),
            dropout=float(self.meta["dropout"]),
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.H = int(self.meta["history"])
        self.id2name: List[str] = list(self.meta.get("id2name", []))
        self.norm = load_norm_stats(norm_stats_json) if norm_stats_json else None

    def _prep(self, states: np.ndarray) -> torch.Tensor:
        """
        states: [D] or [H,D]; 若不足 H 帧会左侧复制第一帧补齐；多于 H 取最后 H 帧
        如果提供了 norm_stats_json，视为 RAW 输入，这里做 z-score；否则视为已归一化输入
        """
        s = np.asarray(states, np.float32)
        if s.ndim == 1:
            s = s[None, :]
        H, D = s.shape
        if H < self.H:
            pad = np.repeat(s[:1], self.H - H, axis=0)
            s = np.concatenate([pad, s], axis=0)
        elif H > self.H:
            s = s[-self.H:, :]

        if self.norm is not None:
            s = maybe_norm(s, self.norm)

        x = s.reshape(1, -1)
        return torch.from_numpy(x).to(self.device)

    @torch.no_grad()
    def predict(self, states: np.ndarray, return_probs: bool=False):
        xb = self._prep(states)
        logits = self.model(xb)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
        idx = int(probs.argmax())
        name = self.id2name[idx] if 0 <= idx < len(self.id2name) else None
        return (idx, name, probs) if return_probs else (idx, name)


def infer_cli(args):
    pred = StagePredictor(args.load_dir, args.norm_stats)

    # 输入：--state（单行）、--state_rows（多行；分号隔行）、--state_file（.npy）
    if args.state_file:
        arr = np.load(args.state_file)  # [D] 或 [H,D]
        states = arr
    elif args.state_rows:
        rows = []
        for row in args.state_rows.split(";"):
            row = row.strip()
            if row:
                rows.append([float(x) for x in row.split(",") if x.strip() != ""])
        states = np.asarray(rows, np.float32)
    elif args.state:
        states = np.asarray([float(x) for x in args.state.split(",") if x.strip() != ""], np.float32)
    else:
        raise ValueError("Provide --state / --state_rows / --state_file")

    if args.probs:
        idx, name, p = pred.predict(states, return_probs=True)
        print(f"stage_idx={idx}  stage_name={name}")
        print("probs=", p.tolist())
    else:
        idx, name = pred.predict(states, return_probs=False)
        print(f"stage_idx={idx}  stage_name={name}")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Train or infer a supervised stage classifier on normalized states.")
    sub = ap.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--states_cache", required=True, help="episode_states_cache.json（已归一化的 state）")
    pt.add_argument("--segments_json", required=True, help="包含 episode_name + segments 的 JSON")
    pt.add_argument("--episodes_json", default=None, help="白名单：只用这里列出的 episode 文件名")
    pt.add_argument("--out_dir", required=True)
    pt.add_argument("--epochs", type=int, default=5)
    pt.add_argument("--bs", type=int, default=512)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--wd", type=float, default=1e-4)
    pt.add_argument("--hidden", default="256,256")
    pt.add_argument("--dropout", type=float, default=0.0)
    pt.add_argument("--history", type=int, default=1)
    pt.add_argument("--use_label_name", action="store_true", help="用 segments.label 作为类别（跨 episode 共享语义）")
    pt.add_argument("--class_weight", action="store_true", help="按 1/freq 做类别加权以缓解不平衡")
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--cpu", action="store_true")

    pi = sub.add_parser("infer")
    pi.add_argument("--load_dir", required=True, help="训练输出目录（含 stage_head.pt）")
    pi.add_argument("--norm_stats", default=None, help="若推理输入是 RAW state，提供训练期 norm_stats.json 做 z-score；如果输入已归一化，留空")
    pi.add_argument("--probs", action="store_true", help="打印全类别概率")
    # 三选一输入
    pi.add_argument("--state", default=None, help="一行 state，用逗号分隔")
    pi.add_argument("--state_rows", default=None, help="多行 state，用分号隔开每一行（历史 H 行）")
    pi.add_argument("--state_file", default=None, help=".npy，形状 [D] 或 [H,D]")

    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        infer_cli(args)

if __name__ == "__main__":
    main()




'''
uv run scripts/train_stage_segmentation.py train \
  --states_cache metadata/libero/episode_states_cache.json \
  --segments_json examples/libero/all_segments_summary.json \
  --episodes_json examples/libero/all_stage_clean.json \
  --out_dir checkpoints/stage_head \
  --epochs 50 --bs 512 --history 4 \
  --use_label_name \
  --class_weight
'''

'''
python scripts/train_or_infer_stage_classifier.py infer \
  --load_dir checkpoints/stage_head \
  --norm_stats metadata/libero/norm_stats.json \
  --state "s0,s1,s2,..."
'''
