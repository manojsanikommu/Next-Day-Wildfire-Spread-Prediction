"""
cnn_baseline.py

Simple CNN baseline for tile-level wildfire detection.
Input:  (12, 64, 64)
Output: scalar probability of "fire tile" (any fire pixel).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import seed_everything


class TileDataset(Dataset):
    def __init__(self, X, Y):
        """
        X: (N, H, W, C)
        Y: (N, H, W, 1) -> converted to tile label
        """
        # Convert to channels-first for CNN
        X_chw = np.transpose(X, (0, 3, 1, 2))  # (N, C, H, W)
        self.X = X_chw.astype(np.float32)

        # Tile label: 1 if any fire pixel
        N = Y.shape[0]
        tile_sum = Y.reshape(N, -1).sum(axis=1)
        self.y = (tile_sum > 0).astype(np.float32)  # shape (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y


class FireTileCNN(nn.Module):
    def __init__(self, in_ch=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)  # (N,)


def main():
    seed_everything(42)

    data_dir = "data"
    outputs_dir = "outputs"
    models_dir = os.path.join(outputs_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("Loading data...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val = np.load(os.path.join(data_dir, "Y_val.npy"))

    # Binarize labels just in case
    Y_train = (Y_train > 0).astype(np.uint8)
    Y_val = (Y_val > 0).astype(np.uint8)

    train_ds = TileDataset(X_train, Y_train)
    val_ds = TileDataset(X_val, Y_val)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = FireTileCNN(in_ch=12).to(device)
    # Fire is rare → higher weight on positive class
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def run_epoch(dl, train=True):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        all_probs = []
        all_labels = []

        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.detach().cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        roc = roc_auc_score(all_labels, all_probs)
        pr = average_precision_score(all_labels, all_probs)
        avg_loss = total_loss / len(dl.dataset)
        return avg_loss, roc, pr

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_roc, tr_pr = run_epoch(train_dl, train=True)
        va_loss, va_roc, va_pr = run_epoch(val_dl, train=False)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f}, train_ROC={tr_roc:.3f}, train_PR={tr_pr:.3f} | "
            f"val_loss={va_loss:.4f}, val_ROC={va_roc:.3f}, val_PR={va_pr:.3f}"
        )

    model_path = os.path.join(models_dir, "cnn_tile_baseline.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved CNN baseline model to: {model_path}")


if __name__ == "__main__":
    main()
