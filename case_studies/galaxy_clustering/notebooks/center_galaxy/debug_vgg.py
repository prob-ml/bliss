#!/usr/bin/env python3
# ============================================================
# Galaxy Tile Binary Classifier (VGG + HardClip + arcsinh + AUC)
# ============================================================

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import precision, recall, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Device setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# Random rotation by right angles
# ============================================================
class RandomRightAngleRotation:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle)


# ============================================================
# VGG Block (2×Conv + BN + ReLU)
# ============================================================
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Dataset
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

        # first 3 channels
        img = data["image"].to(torch.float32)[:3]

        # ----- HARD CLIP -----
        img = torch.clamp(img, min=-10.0, max=10.0)

        # ----- arcsinh stretch -----
        alpha = 0.8
        img = torch.asinh(img / alpha)

        # ----- normalize channelwise to [0,1] -----
        img = (img - img.amin(dim=(1,2), keepdim=True)) / (
               img.amax(dim=(1,2), keepdim=True) - img.amin(dim=(1,2), keepdim=True) + 1e-6
        )

        # ----- shift to [-1,1] -----
        img = (img - 0.5) * 2.0

        label = torch.tensor(float(data["label"]))

        if idx == 0:
            print(f"[DEBUG] {path.name}: min={img.min():.3f}, max={img.max():.3f}, "
                  f"mean={img.mean():.3f}, label={label}")

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
# DataModule (80/10/10 split)
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
        n_val = int(0.1 * n_total)

        random.seed(42)
        random.shuffle(all_files)

        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]

        augment = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRightAngleRotation(),
        ])

        self.train_dataset = TileDataset(train_files, transform=augment)
        self.val_dataset = TileDataset(val_files)
        self.test_dataset = TileDataset(test_files)

        def count_labels(files):
            labels = [torch.load(f)["label"] for f in files]
            return sum(labels), len(labels)

        pos, total = count_labels(train_files)
        print(f"Train positives: {pos}/{total} ({pos/total:.2%})")

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
# Galaxy Tile Classifier (VGG-style)
# ============================================================
class GalaxyTileClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, in_channels=3, base_channels=32):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

        self.encoder = nn.Sequential(
            VGGBlock(in_channels, base_channels),
            nn.MaxPool2d(2),

            VGGBlock(base_channels, base_channels * 2),
            nn.MaxPool2d(2),

            VGGBlock(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),

            VGGBlock(base_channels * 4, base_channels * 8),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(base_channels * 4, 1)
        )

        nn.init.constant_(self.classifier[-1].bias, -1.0)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x).squeeze(1)

    def _step(self, batch, stage):
        imgs, labels = batch
        labels = labels.float()
        logits = self(imgs)
        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        acc = (preds == labels).float().mean()
        prec = precision(preds, labels.int(), task="binary")
        rec = recall(preds, labels.int(), task="binary")
        f1 = f1_score(preds, labels.int(), task="binary")

        # log as before
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_precision", prec, on_epoch=True)
        self.log(f"{stage}_recall", rec, on_epoch=True)
        self.log(f"{stage}_f1", f1, prog_bar=True, on_epoch=True)

        # 👇 Return metrics for test summary table
        if stage == "test":
            return {
                "test_loss": loss,
                "test_acc": acc,
                "test_precision": prec,
                "test_recall": rec,
                "test_f1": f1,
            }

        return loss


    def training_step(self, batch, batch_idx): return self._step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._step(batch, "val")
    def test_step(self, batch, batch_idx): return self._step(batch, "test")

    def predict_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        probs = torch.sigmoid(logits)
        return probs.cpu(), labels.cpu()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.learning_rate,
                                weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                         T_max=30,
                                                         eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": sch}


# ============================================================
# Training routine
# ============================================================
def run_training(data_root, sanity_overfit=False):
    model = GalaxyTileClassifier(learning_rate=1e-4, in_channels=3)
    data_module = TileDataModule(data_root, batch_size=64, num_workers=8)
    data_module.setup()

    if sanity_overfit:
        print("⚠️  Sanity mode on 20 samples")
        data_module.train_dataset = Subset(data_module.train_dataset, range(20))
        data_module.val_dataset = Subset(data_module.val_dataset, range(20))

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=3,
        filename="best-{epoch}-{val_loss:.4f}",
        dirpath="checkpoints_tile/"
    )

    early_stop_cb = EarlyStopping(monitor="val_loss",
                                 patience=5,
                                 mode="min",
                                 verbose=True)

    tb_logger = TensorBoardLogger("tb_logs", name="tile_classifier")
    print(f"TensorBoard logs at: {tb_logger.log_dir}")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=tb_logger,
        precision="32-true",
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        enable_progress_bar=True
    )

    trainer.fit(model, data_module)
    results = trainer.test(model, data_module)
    print("\n━━━━━━━━ Test metrics ━━━━━━━━")
    for k, v in results[0].items():
        print(f"{k:20s} {v:.6f}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # =====================================================
    # ✅ ROC–AUC plot (test)
    # =====================================================
    preds = trainer.predict(model, data_module.test_dataloader())
    probs = torch.cat([p for p, _ in preds]).numpy()
    labels = torch.cat([l for _, l in preds]).numpy()

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.savefig("/home/hughwang/bliss/bliss/case_studies/galaxy_clustering/notebooks/center_galaxy/test_auc.png", dpi=200)
    plt.close()

    print(f"✅ ROC–AUC = {roc_auc:.3f}")
    print("✅ Saved → test_auc.png")
    print("✅ Training finished!")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    data_root = "/scratch/regier_root/regier0/hughwang/remerge_cg/file_data_balanced_35000"
    run_training(data_root, sanity_overfit=False)
