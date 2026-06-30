from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "output" / "datasets" / "cmm_model_training_data.parquet"
FEATURE_COLS_PATH = PROJECT_ROOT / "output" / "datasets" / "cmm_feature_columns.txt"
RETURN_COLS_PATH = PROJECT_ROOT / "output" / "datasets" / "cmm_return_window_columns.txt"
MODEL_DIR = PROJECT_ROOT / "output" / "models" / "cmm"


SEED = 42
INITIAL_TRAIN_MONTHS = 60
VAL_FRACTION = 0.30
TEST_BLOCK_MONTHS = 24
MIN_TEST_BLOCK_MONTHS = 6
N_EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = (128, 64, 32)
DROPOUT = 0.05


class CMMNet(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for out_dim in hidden:
            layers.extend([nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(dropout)])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_hat = self.net(z).squeeze(-1)
        scores = returns * z_hat.unsqueeze(1)
        weights = torch.softmax(scores, dim=1)
        cmm = (weights * returns).sum(dim=1)
        return cmm, z_hat, weights


def cs_zscore_tensor(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def expanding_splits(month_array: list[pd.Timestamp]) -> list[dict]:
    start = INITIAL_TRAIN_MONTHS
    fold = 1
    folds = []
    while start < len(month_array):
        train_val = month_array[:start]
        remaining = len(month_array) - start
        block_size = min(TEST_BLOCK_MONTHS, remaining)
        if block_size < MIN_TEST_BLOCK_MONTHS:
            break
        val_size = max(1, int(len(train_val) * VAL_FRACTION))
        folds.append(
            {
                "fold": fold,
                "train_months": train_val[:-val_size],
                "val_months": train_val[-val_size:],
                "test_months": month_array[start : start + block_size],
            }
        )
        start += block_size
        fold += 1
    return folds


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    feature_cols = FEATURE_COLS_PATH.read_text(encoding="utf-8").splitlines()
    return_cols = RETURN_COLS_PATH.read_text(encoding="utf-8").splitlines()
    id_cols = ["stock_id", "signal_date", "trade_date", "exit_date", "target_1m_ret", "target_1m_ret_cs_z"]
    df = pd.read_parquet(DATA_PATH, columns=id_cols + return_cols + feature_cols)
    for col in ["signal_date", "trade_date", "exit_date"]:
        df[col] = pd.to_datetime(df[col])

    months = sorted(df["signal_date"].unique())
    month_indices = {month: np.flatnonzero(df["signal_date"].to_numpy() == month) for month in months}
    x_ret = df[return_cols].to_numpy(dtype=np.float32)
    x_z = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["target_1m_ret_cs_z"].to_numpy(dtype=np.float32)

    def month_tensors(month):
        idx = month_indices[month]
        return (
            torch.from_numpy(x_z[idx]).to(device),
            torch.from_numpy(x_ret[idx]).to(device),
            torch.from_numpy(y[idx]).to(device),
        )

    def run_epoch(model, month_list, optimizer=None):
        is_train = optimizer is not None
        model.train(is_train)
        losses = []
        shuffled = list(month_list)
        if is_train:
            random.shuffle(shuffled)
        for month in shuffled:
            z, r, target = month_tensors(month)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(is_train):
                pred, _, _ = model(z, r)
                loss = nn.functional.mse_loss(cs_zscore_tensor(pred), target)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    @torch.no_grad()
    def predict_months(model, month_list, split: str, fold: int):
        model.eval()
        pieces = []
        for month in month_list:
            idx = month_indices[month]
            z, r, _ = month_tensors(month)
            pred, z_hat, weights = model(z, r)
            out = df.iloc[idx][id_cols].copy()
            out["cmm_signal"] = pred.detach().cpu().numpy()
            out["cmm_signal_cs_z"] = cs_zscore_tensor(pred).detach().cpu().numpy()
            out["z_hat"] = z_hat.detach().cpu().numpy()
            out["max_weight"] = weights.max(dim=1).values.detach().cpu().numpy()
            out["split"] = split
            out["fold"] = fold
            pieces.append(out)
        return pd.concat(pieces, ignore_index=True)

    all_history = []
    all_predictions = []
    fold_rows = []
    fold_model_dir = MODEL_DIR / "fold_models"
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    last_model = None

    for fold_cfg in expanding_splits(months):
        fold = fold_cfg["fold"]
        print(f"\n===== Fold {fold} =====")
        model = CMMNet(n_features=len(feature_cols)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best_val = np.inf
        best_state = None
        bad_epochs = 0
        for epoch in range(1, N_EPOCHS + 1):
            train_loss = run_epoch(model, fold_cfg["train_months"], optimizer)
            val_loss = run_epoch(model, fold_cfg["val_months"])
            all_history.append({"fold": fold, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"fold {fold:02d} epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"fold {fold:02d} early stop")
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        last_model = model
        torch.save({"state_dict": model.state_dict(), "feature_cols": feature_cols, "return_cols": return_cols}, fold_model_dir / f"cmm_model_fold_{fold:02d}.pt")
        all_predictions.append(predict_months(model, fold_cfg["val_months"], "val", fold))
        all_predictions.append(predict_months(model, fold_cfg["test_months"], "test", fold))
        fold_rows.append(
            {
                "fold": fold,
                "train_months": len(fold_cfg["train_months"]),
                "val_months": len(fold_cfg["val_months"]),
                "test_months": len(fold_cfg["test_months"]),
                "train_start": pd.Timestamp(fold_cfg["train_months"][0]).date(),
                "train_end": pd.Timestamp(fold_cfg["train_months"][-1]).date(),
                "val_start": pd.Timestamp(fold_cfg["val_months"][0]).date(),
                "val_end": pd.Timestamp(fold_cfg["val_months"][-1]).date(),
                "test_start": pd.Timestamp(fold_cfg["test_months"][0]).date(),
                "test_end": pd.Timestamp(fold_cfg["test_months"][-1]).date(),
            }
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(MODEL_DIR / "cmm_folds.csv", index=False)
    pd.DataFrame(all_history).to_csv(MODEL_DIR / "cmm_train_history.csv", index=False)
    pred = pd.concat(all_predictions, ignore_index=True)
    pred.to_parquet(MODEL_DIR / "cmm_predictions.parquet", index=False)
    if last_model is not None:
        torch.save(
            {
                "state_dict": last_model.state_dict(),
                "feature_cols": feature_cols,
                "return_cols": return_cols,
                "note": "Last fold model from expanding-window training. Use fold_models/ for each fold.",
            },
            MODEL_DIR / "cmm_model.pt",
        )
    print(json.dumps({"rows": len(pred), "test_rows": int(pred["split"].eq("test").sum())}, indent=2))


if __name__ == "__main__":
    main()

