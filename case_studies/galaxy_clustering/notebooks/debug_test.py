from matplotlib import pyplot as plt
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import glob

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
        x = self.encoder(x)  # → N×C×96×96
        x = self.ctx(x)  # context stack
        x = self.out_conv(x)  # → N×1×96×96
        # Adaptive average pooling reduces the feature map to a fixed grid size (e.g., 4x4),
        # regardless of the input's spatial dimensions. This differs from standard average pooling,
        # which uses a fixed kernel/stride and produces an output size that depends on the input.
        x = F.adaptive_avg_pool2d(x, (4, 4))  # → N×1×4×4 (matches label grid)
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
        accuracy = (pred == target).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", accuracy, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        # AdamW optimizer with ReduceLROnPlateau scheduler for robust training
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=False
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
            },
        }

class ToyGalaxyClusterDataset(Dataset):
    def __init__(
        self,
        sub_file_paths,
        transform=None,
    ):
        self.sub_file_paths = [
            f for f in sub_file_paths
            if ("_subtile_" in f and f.endswith(".pt"))
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
        subtile_data = self._load_tile(self.sub_file_paths[idx])

        img = subtile_data["images"]  # [4, 3072, 3072]

        # Take first 3 channels only:
        img = img[:3, :, :]  # shape [3, 3072, 3072]

        labels = subtile_data["tile_catalog"]["membership"]
        labels = labels.float()
        labels = labels.squeeze()

        # Ensure shape is [1, 4, 4]
        if labels.ndim == 2:
            labels = labels.unsqueeze(0)

        return img, labels



class GalaxyClusterDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size=1, num_workers=2
    ):  # Reduced batch size for large images
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        all_files = glob.glob("/nfs/turbo/lsa-regier/scratch/hughwang/desdir/file_data_bw/*.pt")
        n_total = len(all_files)
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
        self.train_dataset = ToyGalaxyClusterDataset(train_files)
        self.val_dataset   = ToyGalaxyClusterDataset(val_files)
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
        print(label)
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
    data_module = GalaxyClusterDataModule(batch_size=1, num_workers=8)
    data_module.setup()  # Ensure datasets are initialized
    plot_sample_images(data_module.train_dataset, num_samples=3)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices="auto",
        callbacks=[],  # No callbacks
        logger=False,  # Disable logging
        log_every_n_steps=10,
        precision="32-true",  # could use 16-mixed precision to save memory
        gradient_clip_val=1.0,  # Gradient clipping
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,  # Disable checkpointing
        num_sanity_val_steps=2,
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


run_training()
