#!/usr/bin/env python3
import click

from bliss.encoders.detection import DetectionEncoder
from experiment.run.training_functions import run_encoder_training


@click.command()
@click.option("-s", "--seed", required=True, type=int)
@click.option("--train-file", required=True, type=str)
@click.option("--val-file", required=True, type=str)
@click.option("-b", "--batch-size", default=32)
@click.option("-e", "--n-epochs", default=25)
@click.option("--validate-every-n-epoch", default=1, type=int)
@click.option("--val-check-interval", default=0.15, type=float, help="Fraction of training epoch")
@click.option("--log-every-n-steps", default=16, type=int)
def main(
    seed: int,
    train_file: str,
    val_file: str,
    batch_size: int,
    n_epochs: int,
    validate_every_n_epoch: int,
    val_check_interval: int,
    log_every_n_steps: int,
):
    model = DetectionEncoder()
    run_encoder_training(
        seed=seed,
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        n_epochs=n_epochs,
        model=model,
        model_name="detection",
        validate_every_n_epoch=validate_every_n_epoch,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
    )


if __name__ == "__main__":
    main()
