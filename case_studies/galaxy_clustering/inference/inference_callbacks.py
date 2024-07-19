import os

import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class DESPredictionsWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        name = f"rank_{trainer.global_rank}_batchIdx_{batch_idx}_dataloaderIdx_{dataloader_idx}.pt"
        torch.save(
            prediction,
            os.path.join(
                self.output_dir,
                name,
            ),
        )

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(
            batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt")
        )
