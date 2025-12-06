"""
train_unet_small.py

Train a small UNet on 64x64 wildfire tiles:
    Input:  (12, 64, 64)
    Output: (1, 64, 64) binary mask for next-day fire.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import seed_everything, WildfireTileDataset, bce_dice_loss, dice_coefficient


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_ch=12, base_ch=16):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)          # 12 -> 16
        self.pool1 = nn.MaxPool2d(2)                    # 64 -> 32
        self.enc2 = DoubleConv(base_ch, base_ch * 2)    # 16 -> 32
        self.pool2 = nn.MaxPool2d(2)                    # 32 -> 16

        self.bottleneck = DoubleConv(base_ch * 2, base_ch * 4)  # 32 -> 64

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)        # concat skip -> 64 channels

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        out = self.out_conv(d1)
        return out  # logits (N, 1, 64, 64)


def load_numpy(data_dir: str = "data"):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))  # (N, 64, 64, 12)
    Y_train = np.load(os.path.join(data_dir, "Y_train.npy"))  # (N, 64, 64, 1)
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val = np.load(os.path.join(data_dir, "Y_val.npy"))

    # Ensure binary labels (0/1)
    Y_train = (Y_train > 0).astype(np.uint8)
    Y_val = (Y_val > 0).astype(np.uint8)

    # Transpose to (N, C, H, W)
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_val = np.transpose(X_val, (0, 3, 1, 2))

    # Also transpose Y to (N, 1, H, W)
    Y_train = np.transpose(Y_train, (0, 3, 1, 2))
    Y_val = np.transpose(Y_val, (0, 3, 1, 2))

    # Normalize X globally: (x - mean) / std
    mean = X_train.mean()
    std = X_train.std() + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    print("Shapes after transpose:")
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_val  :", X_val.shape, "Y_val  :", Y_val.shape)

    return X_train, Y_train, X_val, Y_val


def main():
    seed_everything(42)

    data_dir = "data"
    outputs_dir = "outputs"
    models_dir = os.path.join(outputs_dir, "models")
    curves_dir = os.path.join(outputs_dir, "curves")
    examples_dir = os.path.join(outputs_dir, "examples")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(examples_dir, exist_ok=True)

    print("Loading NumPy arrays...")
    X_train, Y_train, X_val, Y_val = load_numpy(data_dir)

    train_ds = WildfireTileDataset(X_train, Y_train)
    val_ds = WildfireTileDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = SmallUNet(in_ch=12, base_ch=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 40
    train_loss_hist, val_loss_hist = [], []
    train_dice_hist, val_dice_hist = [], []

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = bce_dice_loss(logits, yb, bce_weight=0.5)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            running_dice += dice_coefficient(logits.detach(), yb.detach()) * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_dice = running_dice / len(train_loader.dataset)
        train_loss_hist.append(train_loss)
        train_dice_hist.append(train_dice)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = bce_dice_loss(logits, yb, bce_weight=0.5)

                val_loss += loss.item() * xb.size(0)
                val_dice += dice_coefficient(logits, yb) * xb.size(0)

        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        val_loss_hist.append(val_loss)
        val_dice_hist.append(val_dice)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss={train_loss:.4f}, Dice={train_dice:.4f} | "
            f"Val Loss={val_loss:.4f}, Dice={val_dice:.4f}"
        )

    # Save model
    model_path = os.path.join(models_dir, "unet_small_wildfire.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Optionally save curves as numpy for plotting later
    np.save(os.path.join(curves_dir, "unet_small_train_loss.npy"), np.array(train_loss_hist))
    np.save(os.path.join(curves_dir, "unet_small_val_loss.npy"), np.array(val_loss_hist))
    np.save(os.path.join(curves_dir, "unet_small_train_dice.npy"), np.array(train_dice_hist))
    np.save(os.path.join(curves_dir, "unet_small_val_dice.npy"), np.array(val_dice_hist))


if __name__ == "__main__":
    main()
