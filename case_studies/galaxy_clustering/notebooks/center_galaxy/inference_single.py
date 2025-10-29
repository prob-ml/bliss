#!/usr/bin/env python3
# ============================================================
# Galaxy Tile Binary Classifier (True/False per 256×256 tile)
# Training + Inference (False Positives + ROC-AUC)
# ============================================================

from matplotlib import pyplot as plt
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from torchmetrics.functional import precision, recall, f1_score
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


# ============================================================
# Device setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import torchvision.transforms.functional as TF

class RandomRightAngleRotation:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle)


# ============================================================
# ConvBlock
# ============================================================
class ConvBlock(nn.Module):
    """Conv2d + GroupNorm + ReLU (+ optional dropout)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, groups=8, dilation=1, dropout_p=None):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p:
            layers.append(nn.Dropout2d(dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ============================================================
# Dataset class
# ============================================================
class TileDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = [f for f in file_paths if f.suffix == ".pt"]
        if len(self.file_paths) == 0:
            raise ValueError("No .pt tiles found!")
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = torch.load(path, map_location="cpu")
        img = data["image"].float()[:3, :, :]        # [3,256,256]
        label = torch.tensor(float(data["label"]))   # True→1.0, False→0.0
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# DataModule
# ============================================================
class TileDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=64, num_workers=8):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        all_files = list(self.data_root.glob("*.pt"))
        n_total = len(all_files)
        print(f"Found {n_total} tiles")

        n_train = int(0.8 * n_total)
        n_val   = int(0.1 * n_total)
        random.seed(42)
        random.shuffle(all_files)

        train_files = all_files[:n_train]
        val_files   = all_files[n_train : n_train + n_val]
        test_files  = all_files[n_train + n_val :]

        augment = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRightAngleRotation(),
            T.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])

        self.train_dataset = TileDataset(train_files, transform=augment)
        self.val_dataset   = TileDataset(val_files)
        self.test_dataset  = TileDataset(test_files)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)


# ============================================================
# Model
# ============================================================
class GalaxyTileClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, in_channels=3, base_channels=32, dropout_p=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels, 3, 2, 1, dropout_p=dropout_p),       # 128×128
            ConvBlock(base_channels, base_channels*2, 3, 2, 1, dropout_p=dropout_p),   # 64×64
            ConvBlock(base_channels*2, base_channels*4, 3, 2, 1, dropout_p=dropout_p), # 32×32
            ConvBlock(base_channels*4, base_channels*8, 3, 2, 1, dropout_p=dropout_p), # 16×16
        )

        self.context = nn.Sequential(
            ConvBlock(base_channels*8, base_channels*8, 3, 1, padding=2, dilation=2, dropout_p=dropout_p),
            ConvBlock(base_channels*8, base_channels*8, 3, 1, padding=4, dilation=4, dropout_p=dropout_p),
            ConvBlock(base_channels*8, base_channels*8, 3, 1, padding=8, dilation=8, dropout_p=dropout_p),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels*8, base_channels*4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(base_channels*4, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.context(x)
        return self.classifier(x).squeeze(1)

    def _step(self, batch, stage):
        imgs, labels = batch
        labels = labels.to(torch.float32)
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        preds = (torch.sigmoid(logits) > 0.5).float()

        acc  = (preds == labels).float().mean()
        prec = precision(preds, labels.int(), task="binary")
        rec  = recall(preds, labels.int(), task="binary")
        f1   = f1_score(preds, labels.int(), task="binary")

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        self.log(f"{stage}_precision", prec)
        self.log(f"{stage}_recall", rec)
        self.log(f"{stage}_f1", f1, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx): return self._step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._step(batch, "val")
    def test_step(self, batch, batch_idx): return self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": sch}


# ============================================================
# Training routine
# ============================================================
def run_training():
    data_root = "/scratch/regier_root/regier0/hughwang/remerge_cg/file_data_balanced_35000"
    model = GalaxyTileClassifier(learning_rate=1e-3, in_channels=3)
    data_module = TileDataModule(data_root, batch_size=64, num_workers=8)
    data_module.setup()

    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3,
                                    filename="best-{epoch}-{val_loss:.4f}",
                                    dirpath="checkpoints_tile/")
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    tb_logger = TensorBoardLogger("tb_logs", name="tile_classifier")
    print(f"TensorBoard logs at: {tb_logger.log_dir}")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=tb_logger,
        precision="32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    print("✅ Training finished!")


# ============================================================
# Inference on TEST set only (False Positives + ROC AUC)
# ============================================================
def run_inference():
    data_root = "/scratch/regier_root/regier0/hughwang/remerge_cg/file_data_balanced_35000"
    checkpoint_path = "/home/hughwang/bliss/bliss/checkpoints_tile/best-epoch=13-val_loss=0.6931.ckpt"
    output_csv = "/home/hughwang/bliss/bliss/case_studies/galaxy_clustering/notebooks/center_galaxy/false_positives.csv"
    roc_path = "/home/hughwang/bliss/bliss/case_studies/galaxy_clustering/notebooks/center_galaxy/roc_curve.png"

    print(f"Using device: {device}")
    print(f"Loading model from {checkpoint_path}")
    model = GalaxyTileClassifier.load_from_checkpoint(checkpoint_path)
    model.eval().to(device)

    # --- Load only TEST split ---
    data_module = TileDataModule(data_root, batch_size=64, num_workers=8)
    data_module.setup()
    loader = data_module.test_dataloader()
    dataset = data_module.test_dataset
    print(f"🧩 Test set size: {len(dataset)}")

    # --- Inference ---
    false_positives = []
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="Inference on test set")):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            for i in range(len(preds)):
                if preds[i] == 1.0 and labels[i] == 0.0:
                    false_positives.append({
                        "file": str(dataset.file_paths[batch_idx * loader.batch_size + i]),
                        "probability": float(probs[i].cpu())
                    })

    # --- Save CSV ---
    if len(false_positives) > 0:
        pd.DataFrame(false_positives).to_csv(output_csv, index=False)
        print(f"✅ Saved {len(false_positives)} false positives to {output_csv}")
    else:
        print("🎯 No false positives found!")

    # --- ROC-AUC ---
    print("\n📈 Computing ROC-AUC curve...")
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_value = roc_auc_score(all_labels, all_probs)
    print(f"✅ ROC AUC = {auc_value:.4f}")

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Galaxy Tile Classifier (Test Set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"📊 ROC curve saved to: {roc_path}")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    # run_training()      # Uncomment to train
    run_inference()       # Only runs on 10% test set
