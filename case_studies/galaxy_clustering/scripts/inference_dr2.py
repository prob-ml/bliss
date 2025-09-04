from matplotlib import pyplot as plt
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import glob
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score
import zipfile
import random
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.transforms.functional as TF
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ConvBlock(nn.Module):
    """
    A block consisting of Conv2d, GroupNorm, and ReLU.
    Optionally supports dilation and dropout.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=8,
        dilation=1,
        dropout_p=None,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(),
        ]
        if dropout_p is not None:
            layers.append(nn.Dropout2d(dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class GalaxyClusterFinder(pl.LightningModule):
    """
    PyTorch Lightning module for detecting galaxy clusters in large synthetic images.

    Architecture:
    - Deep encoder with aggressive downsampling for very large (e.g., 3072x3072) images.
    - Context module with dilated convolutions to capture large-scale spatial context.
    - Output head reduces the feature map to a coarse grid (e.g., 4x4), predicting cluster
      presence per tile.
    - Uses BCEWithLogitsLoss for multi-label binary classification (cluster/no cluster per tile).

    Designed for efficient, coarse segmentation of large astronomical images, where each grid
    cell (tile) is classified as containing a galaxy cluster or not.
    """

    def __init__(
        self,
        learning_rate=1e-3,
        in_channels=3,
        base_channels=32,
        groups=8,
        dropout_p=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()

        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()

        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()

        # Encoder: progressively downsamples the input image to a compact feature map.
        # Each block halves the spatial resolution and increases the channel count.
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels, 3, 2, 1, groups=groups),
            ConvBlock(base_channels, base_channels * 2, 3, 2, 1, groups=groups),
            ConvBlock(base_channels * 2, base_channels * 4, 3, 2, 1, groups=groups),
            ConvBlock(base_channels * 4, base_channels * 8, 3, 2, 1, groups=groups),
            ConvBlock(base_channels * 8, base_channels * 8, 3, 2, 1, groups=groups),
        )

        # Context module: stack of dilated convolutions to provide a large receptive field.
        # Dilated convolutions allow each output pixel to aggregate information from a much wider area
        # of the input feature map, without increasing the number of parameters or reducing resolution.
        # This is crucial for detecting large-scale structures (e.g., galaxy clusters) that may span
        # many tiles. Here, we use dilation rates of 2, 4, and 8 to exponentially increase the receptive
        # field, enabling the model to "see" global context efficiently.
        ctx = []
        for i in range(3):
            d = 2 ** (i + 1)
            dilated_cb = ConvBlock(
                base_channels * 8, base_channels * 8, 3, 1, d, groups, d, dropout_p
            )
            ctx.append(dilated_cb)
        self.ctx = nn.Sequential(*ctx)

        self.pre_downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        # Output head: reduces channels and produces a single-channel (logit) output per tile.
        # The final adaptive average pooling reduces the output to a fixed grid (e.g., 4x4),
        # matching the label format for cluster presence per tile.
        self.out_conv = nn.Sequential(
            ConvBlock(
                base_channels * 8,
                base_channels * 4,
                1,
                1,
                0,
                groups=groups,
                dropout_p=None,
            ),
            nn.Conv2d(base_channels * 4, 1, kernel_size=1),
        )

    def forward(self, x):
        # Pass input through encoder, context, and output head
        x = self.pre_downsample(x)  # Downsample to 4864x4864 before encoding
        x = self.encoder(x)  # → N×C×96×96
        x = self.ctx(x)  # context stack
        x = self.out_conv(x)  # → N×1×96×96
        # Adaptive average pooling reduces the feature map to a fixed grid size (e.g., 10x10),
        # regardless of the input's spatial dimensions. This differs from standard average pooling,
        # which uses a fixed kernel/stride and produces an output size that depends on the input.
        x = F.adaptive_avg_pool2d(x, (19, 19))  # → N×1×19×19 (matches label grid)
        return x  # Elide sigmoid since we're using BCEWithLogitsLoss

    def _generic_step(self, batch, batch_idx, stage):
        """
        Shared logic for training, validation, and test steps.
        Computes loss and accuracy for the current batch.
        """
        img, target = batch
        output = self(img)
        loss = self.criterion(output, target)

        pred = (torch.sigmoid(output) > 0.5).float()
        # print(f"Batch target sum: {target.sum().item()}")
        # print(f"Pred positives: {pred.sum().item()}")
        accuracy = (pred == target).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_acc", accuracy, prog_bar=True, sync_dist=True)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1).int()

        if stage == "train":
            self.train_precision.update(pred_flat, target_flat)
            self.train_recall.update(pred_flat, target_flat)
            self.train_f1.update(pred_flat, target_flat)
        elif stage == "val":
            self.val_precision.update(pred_flat, target_flat)
            self.val_recall.update(pred_flat, target_flat)
            self.val_f1.update(pred_flat, target_flat)
        elif stage == "test":
            self.test_precision.update(pred_flat, target_flat)
            self.test_recall.update(pred_flat, target_flat)
            self.test_f1.update(pred_flat, target_flat)

        return loss

    def training_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        self.log("train_precision", self.train_precision.compute(), prog_bar=True, sync_dist=True)
        self.log("train_recall", self.train_recall.compute(), prog_bar=True, sync_dist=True)
        self.log("train_f1", self.train_f1.compute(), prog_bar=True, sync_dist=True)
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.log("val_precision", self.val_precision.compute(), prog_bar=True, sync_dist=True)
        self.log("val_recall", self.val_recall.compute(), prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True, sync_dist=True)
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):
        self.log("test_precision", self.test_precision.compute(), prog_bar=True, sync_dist=True)
        self.log("test_recall", self.test_recall.compute(), prog_bar=True, sync_dist=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True, sync_dist=True)
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,  # Total epochs or steps per cycle
            eta_min=1e-6  # Minimum LR
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Step scheduler per epoch
                "frequency": 1
            }
        }



class RotateFlipTransform:
    def __init__(self):
        # You can customize the set of possible angles and flips
        self.angles = [0, 90, 180, 270]
        self.do_hflip = True
        self.do_vflip = True

    def __call__(self, sample):
        img = sample["images"]
        tile_labels = sample["tile_catalog"]["membership"]

        # print(tile_labels.shape)

        # Random rotation
        angle = random.choice(self.angles)
        img = TF.rotate(img, angle)
        tile_labels = self.rotate_tensor(tile_labels, angle)

        # Random horizontal flip
        if self.do_hflip and random.random() > 0.5:
            img = TF.hflip(img)
            tile_labels = tile_labels.flip(-1)  # horizontal = width

        # Random vertical flip
        if self.do_vflip and random.random() > 0.5:
            img = TF.vflip(img)
            tile_labels = tile_labels.flip(-2)  # vertical = height

        sample["images"] = img
        sample["tile_catalog"]["membership"] = tile_labels
        return sample

    def rotate_tensor(self, tensor, angle):
        if tensor.ndim == 4:  # shape [1, 4, 4, 1] or similar
            tensor = tensor.squeeze(-1).squeeze(0)

        if angle == 90:
            return tensor.permute(1, 0).flip(0).unsqueeze(0)  # transpose + vertical flip
        elif angle == 180:
            return tensor.flip(0).flip(1).unsqueeze(0)
        elif angle == 270:
            return tensor.permute(1, 0).flip(1).unsqueeze(0)  # transpose + horizontal flip

        return tensor.unsqueeze(0)




class ToyGalaxyClusterDataset(Dataset):
    def __init__(
        self,
        sub_file_paths,
        transform=None,
    ):
        self.sub_file_paths = [
            f for f in sub_file_paths
            if (f.endswith(".pt"))
        ]

        if len(self.sub_file_paths) == 0:
            raise ValueError("No subtile .pt files found in provided paths!")
        self.transform = transform

    def _load_tile(self, path):
        with open(path, "rb") as f:
            data = torch.load(f, map_location="cpu")
        return self.transform(data) if self.transform else data

    def __len__(self):
        return len(self.sub_file_paths)

    def __getitem__(self, idx):
        try:
            subtile_data = self._load_tile(self.sub_file_paths[idx])
        except Exception:
            print(f"Error loading subtile {self.sub_file_paths[idx]}, skipping...")
            return self.__getitem__((idx + 1) % len(self.sub_file_paths))

        # if self.transform:
        #     subtile_data = self.transform(subtile_data)

        img = subtile_data["images"]  # [4, H, W]
        img = img[:3, :, :]           # → [3, H, W]

        labels = subtile_data["tile_catalog"]["membership"]
        labels = labels.float().squeeze()

        if labels.ndim == 2:
            labels = labels.unsqueeze(0)  # Ensure [1, 4, 4]

        return img, labels



class GalaxyClusterDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size=1, num_workers=2
    ):  # Reduced batch size for large images
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # all_files = glob.glob("/scratch/regier_root/regier0/hughwang/512combined/file_data_*.pt")
        all_files = glob.glob("/scratch/regier_root/regier0/hughwang/realCombined/file_data_*.pt")
        # all_files = glob.glob("/nfs/turbo/lsa-regier/scratch/gapatron/galaxy_clustering/desdr1_galsim/file_data/*.pt")
        # all_files = glob.glob("/nfs/turbo/lsa-regier/scratch/hughwang/desdir/file_data/*.pt")[:2000]
        n_total = len(all_files)
        print(n_total)
        n_train = int(0.8 * n_total)
        n_val   = int(0.1 * n_total)
        n_test  = n_total - n_train - n_val

        # Shuffle the file list reproducibly
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(n_total, generator=g).tolist()
        shuffled_files = [all_files[i] for i in perm]

        train_files = shuffled_files[:n_train]
        val_files   = shuffled_files[n_train : n_train + n_val]
        test_files  = shuffled_files[n_train + n_val :]

        # Create 3 *separate* dataset objects
        transform = RotateFlipTransform()

        self.train_dataset = ToyGalaxyClusterDataset(train_files, transform=transform)
        self.val_dataset   = ToyGalaxyClusterDataset(val_files)  # No augmentation on val/test
        self.test_dataset  = ToyGalaxyClusterDataset(test_files)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# Plot and save a few sample images from the dataset
def plot_sample_images(dataset, num_samples=3):
    for i in range(num_samples):
        print(f"Sample {i}")
        img, label = dataset[i]
        print(f"Label sum: {label.sum().item()}, Has cluster: {label.sum().item() > 0}")
        plt.figure(figsize=(6, 6))
        img_np = img.cpu().permute(1, 2, 0).numpy()
        plt.imshow(img_np)
        ax = plt.gca()
        # Draw grid lines
        grid_n = label.shape[-1]
        img_size = img.shape[1]
        tile = img_size // grid_n
        for t in range(1, grid_n):
            ax.axhline(t * tile, color="white", lw=1, ls="--", alpha=0.7)
            ax.axvline(t * tile, color="white", lw=1, ls="--", alpha=0.7)
        # Highlight labeled tiles
        for row in range(grid_n):
            for col in range(grid_n):
                if label[0, row, col] > 0.5:
                    rect = plt.Rectangle(
                        (col * tile, row * tile),
                        tile,
                        tile,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="red",
                        alpha=0.25,
                    )
                    ax.add_patch(rect)
        plt.title(f"Sample {i}")
        plt.axis("off")
        plt.savefig(f"sample_{i}.png", bbox_inches="tight")
        plt.close()


def run_training():
    model = GalaxyClusterFinder(learning_rate=1e-3)
    data_module = GalaxyClusterDataModule(batch_size=2, num_workers=3)
    data_module.setup()
    plot_sample_images(data_module.train_dataset, num_samples=3)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",        # what to monitor
        mode="min",                # lower is better
        save_top_k=5,              # keep only the best one so far
        filename="best-{epoch}-{val_loss:.4f}",
        dirpath="checkpoints_731_real/",    # where to save
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",    # Metric to monitor
        patience=5,            # Number of epochs with no improvement after which training stops
        mode="min",            # Minimize the monitored metric
        verbose=True           # Optional: logs info when triggered
    )

    tb_logger = TensorBoardLogger("tb_logs", name="galaxy_cluster")
    print(f"Logging to TensorBoard at: {tb_logger.log_dir}")

    trainer = Trainer(
        max_epochs=50,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_cb],
        logger=tb_logger,
        log_every_n_steps=10,
        precision="32-true",
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=True,  # optional — True by default if using ModelCheckpoint
        num_sanity_val_steps=2,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)

import glob
import torch

all_files = glob.glob("/scratch/regier_root/regier0/hughwang/realCombined_dr2/file_data_*.pt")
n_total = len(all_files)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)

# Reproducible shuffle
g = torch.Generator().manual_seed(42)
perm = torch.randperm(n_total, generator=g).tolist()
shuffled_files = [all_files[i] for i in perm]

test_files = shuffled_files[n_train + n_val:]
print(len(test_files), "test files found")


test_dataset = ToyGalaxyClusterDataset(test_files)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

model = GalaxyClusterFinder.load_from_checkpoint("/home/hughwang/bliss/bliss/checkpoints_731_real/best-epoch=49-val_loss=0.4247.ckpt")
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def collect_scores_and_labels(model, loader, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    all_scores = []
    all_labels = []
    for imgs, targets in loader:
        imgs = imgs.to(device)               # [B, 3, H, W]
        targets = targets.to(device).float() # [B, 1, 19, 19]

        logits = model(imgs)                 # [B, 1, 19, 19]
        probs = torch.sigmoid(logits)        # same shape

        all_scores.append(probs.detach().cpu().numpy().ravel())
        all_labels.append(targets.detach().cpu().numpy().ravel())

    y_score = np.concatenate(all_scores, axis=0)  # shape [N_pixels]
    y_true  = np.concatenate(all_labels, axis=0)  # shape [N_pixels]
    return y_true, y_score

# --- collect
device = next(model.parameters()).device
y_true, y_score = collect_scores_and_labels(model, test_loader, device)

# --- ROC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# --- plot & save
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], lw=1, ls="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True, ls="--", alpha=0.6)
os.makedirs("predictions/metrics", exist_ok=True)
plt.savefig("predictions/metrics/roc_curve.png", bbox_inches="tight", dpi=150)
plt.close()

# (optional) also save raw points
np.savez("predictions/metrics/roc_points.npz", fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc)
print(f"ROC AUC: {roc_auc:.4f}, saved to predictions/metrics/")






# import torch
# from matplotlib import pyplot as plt

# wrong_samples = []

# with torch.no_grad():
#     for idx, (img, target) in enumerate(test_loader):
#         img = img.to(model.device)
#         target = target.to(model.device)

#         output = model(img)
#         pred = (torch.sigmoid(output) > 0.5).float()

#         if (pred != target).sum() > 0:
#             # Include the corresponding subtile file path
#             wrong_samples.append((img.cpu(), pred.cpu(), target.cpu(), test_files[idx]))

#         if len(wrong_samples) >= 5:
#             break


# from astropy.visualization import make_lupton_rgb
# import matplotlib.pyplot as plt

# for i, (img, pred, target, path) in enumerate(wrong_samples):
#     img = img.squeeze(0).to(model.device)       # [3, H, W]
#     target = target.squeeze().to(model.device)  # [1, 4, 4] or [4, 4]

#     # Get probability (before thresholding)
#     with torch.no_grad():
#         output = model(img.unsqueeze(0))  # add batch dim → [1, 1, 4, 4]
#         probs = torch.sigmoid(output).squeeze().cpu()  # [4, 4]

#     # Convert to numpy bands for Lupton
#     g_band = img[0].cpu().numpy()
#     r_band = img[1].cpu().numpy()
#     i_band = img[2].cpu().numpy()
#     rgb = make_lupton_rgb(i_band, r_band, g_band)

#     # Plot
#     plt.figure(figsize=(20, 20))
#     plt.imshow(rgb, origin='upper')
#     ax = plt.gca()

#     grid_n = target.shape[-1]
#     img_size = img.shape[1]
#     tile = img_size // grid_n

#     for row in range(grid_n):
#         for col in range(grid_n):
#             x0, y0 = col * tile, row * tile
#             gt = target[row, col].item()
#             pval = probs[row, col].item()

#             # Draw boxes
#             if gt > 0.5:
#                 ax.add_patch(plt.Rectangle((x0, y0), tile, tile, edgecolor="green", facecolor="none", linewidth=1.5))
#             if pval > 0.5 and gt <= 0.5:
#                 ax.add_patch(plt.Rectangle((x0, y0), tile, tile, edgecolor="red", facecolor="none", linestyle="--", linewidth=1.5))

#             # Annotate confidence
#             cx, cy = x0 + tile // 2, y0 + tile // 2
#             ax.text(cx, cy, f"{pval:.2f}", color="white", fontsize=8, ha="center", va="center",
#                     bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"))

#     plt.title(f"Wrong Sample {i}")
#     plt.axis("off")
#     filename = os.path.basename(path).replace(".pt", "")
#     os.makedirs("predictions_dr2/wrong", exist_ok=True)  # <-- Add this line
#     plt.savefig(f"predictions_dr2/wrong/{filename}_lupton.png", bbox_inches="tight")
#     plt.close()


# correct_samples = []

# with torch.no_grad():
#     for idx, (img, target) in enumerate(test_loader):
#         img = img.to(model.device)
#         target = target.to(model.device)

#         output = model(img)
#         pred = (torch.sigmoid(output) > 0.5).float()

#         match_ratio = (pred == target).float().mean().item()
#         true_positives = ((pred == 1) & (target == 1)).sum().item()

#         # ≥90% correct & at least one true positive
#         if match_ratio >= 0.85 and true_positives > 0:
#             correct_samples.append((img.cpu(), pred.cpu(), target.cpu(), test_files[idx]))

#         if len(correct_samples) >= 10:
#             break

# print(len(correct_samples))

# # Plot and save
# for i, (img, pred, target, path) in enumerate(correct_samples):
#     img = img.squeeze(0).to(model.device)       # [3, H, W]
#     target = target.squeeze().to(model.device)  # [1, 4, 4] or [4, 4]

#     # Get probability map (before thresholding)
#     with torch.no_grad():
#         output = model(img.unsqueeze(0))  # add batch dim
#         probs = torch.sigmoid(output).squeeze().cpu()

#     # Lupton RGB conversion
#     g_band = img[0].cpu().numpy()
#     r_band = img[1].cpu().numpy()
#     i_band = img[2].cpu().numpy()
#     rgb = make_lupton_rgb(i_band, r_band, g_band)

#     plt.figure(figsize=(20, 20))
#     plt.imshow(rgb, origin='upper')
#     ax = plt.gca()

#     grid_n = target.shape[-1]
#     img_size = img.shape[1]
#     tile = img_size // grid_n

#     for row in range(grid_n):
#         for col in range(grid_n):
#             x0, y0 = col * tile, row * tile
#             gt = target[row, col].item()
#             pval = probs[row, col].item()

#             # Green box for GT clusters
#             if gt > 0.5:
#                 ax.add_patch(plt.Rectangle((x0, y0), tile, tile, edgecolor="green",
#                                            facecolor="none", linewidth=1.5))
#             # Annotate confidence
#             cx, cy = x0 + tile // 2, y0 + tile // 2
#             ax.text(cx, cy, f"{pval:.2f}", color="white", fontsize=8, ha="center", va="center",
#                     bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"))
#     plt.axis("off")

#     filename = os.path.basename(path).replace(".pt", "")
#     os.makedirs("predictions_dr2/correct_p", exist_ok=True)
#     plt.savefig(f"predictions_dr2/correct_p/{filename}_lupton.png", bbox_inches="tight")
#     plt.close()