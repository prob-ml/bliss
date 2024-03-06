import torch

import bliss.simulator.toy_simulated_dataset


def test_toy_simulator():
    simulator = bliss.simulator.toy_simulated_dataset.ToySimulator(height=80, width=80, n_bands=2)

    ds = bliss.simulator.toy_simulated_dataset.ToySimulatedDataset(simulator, batch_size=64)
    batch = next(iter(ds))
    assert batch["images"].shape == (64, 2, 80, 80)

    ds = bliss.simulator.toy_simulated_dataset.ToySimulatedDataset(
        simulator, batch_size=1, epoch_size=1, cache=True
    )
    batch1 = next(iter(ds))
    batch2 = next(iter(ds))
    assert batch1["images"].shape == (1, 2, 80, 80)
    assert torch.all(batch1["images"] == batch2["images"])
