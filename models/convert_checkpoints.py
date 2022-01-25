import torch

from bliss.models.galaxy_net import OneCenteredGalaxyAE
from bliss.models.binary import BinaryEncoder
from bliss.models.galaxy_encoder import GalaxyEncoder
from bliss.sleep import SleepPhase

models = [OneCenteredGalaxyAE, BinaryEncoder, GalaxyEncoder, SleepPhase, GalaxyEncoder]
checkpoints = ["sdss_autoencoder", "sdss_binary", "sdss_galaxy_encoder", "sdss_sleep", "sdss_galaxy_encoder_real"]

for model, checkpoint in zip(models, checkpoints):
    m = model.load_from_checkpoint("models/" + checkpoint + ".ckpt")
    torch.save(m.state_dict(), "models/" + checkpoint + ".pt")
