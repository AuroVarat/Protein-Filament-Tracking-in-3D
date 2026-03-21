from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import read_tiff


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise ImportError("PyTorch is required only for the optional baseline training hook.") from exc

    return torch, nn, F, DataLoader, Dataset


def train_baseline(synth_dir: Path, output_dir: Path, epochs: int = 5, batch_size: int = 8, lr: float = 1e-3) -> None:
    torch, nn, F, DataLoader, Dataset = _require_torch()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(synth_dir / "manifest.csv")

    class SynthDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, idx: int):
            row = self.frame.iloc[idx]
            image = read_tiff(Path(row.image_path)).astype(np.float32)
            target = read_tiff(Path(row.filament_mask_path)).astype(np.float32)
            image = (image - image.mean()) / max(image.std(), 1e-6)
            return torch.from_numpy(image[None]), torch.from_numpy(target[None])

    class ConvBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.seq(x)

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = ConvBlock(1, 16)
            self.enc2 = ConvBlock(16, 32)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(32, 64)
            self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec2 = ConvBlock(64, 32)
            self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dec1 = ConvBlock(32, 16)
            self.out = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            b = self.bottleneck(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)

    def dice_bce_loss(logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()
        return bce + dice

    train_df = manifest[manifest["split"] == "train"] if "split" in manifest else manifest
    val_df = manifest[manifest["split"] == "val"] if "split" in manifest and (manifest["split"] == "val").any() else train_df.iloc[: max(1, len(train_df) // 5)]

    train_loader = DataLoader(SynthDataset(train_df), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SynthDataset(val_df), batch_size=batch_size, shuffle=False)

    model = SmallUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for images, targets in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = dice_bce_loss(logits, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        example_pred = None
        with torch.no_grad():
            for images, targets in val_loader:
                logits = model(images)
                loss = dice_bce_loss(logits, targets)
                val_losses.append(float(loss.item()))
                if example_pred is None:
                    example_pred = torch.sigmoid(logits[0, 0]).cpu().numpy()

        epoch_record = {"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(epoch_record)
        if epoch_record["val_loss"] < best_val:
            best_val = epoch_record["val_loss"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            if example_pred is not None:
                np.save(output_dir / "example_prediction.npy", example_pred)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
