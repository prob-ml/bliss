import warnings

from bliss.simulator.simulated_dataset import SimulatedDataset
from bliss.surveys.survey import Survey
from case_studies.weak_lensing.lensing_decoder import LensingDecoder
from case_studies.weak_lensing.lensing_prior import LensingPrior

# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


class LensingSimulatedDataset(SimulatedDataset):
    def __init__(
        self,
        *args,
        survey: Survey,
        prior: LensingPrior,
        **kwargs,
    ):
        super().__init__(*args, survey, prior, **kwargs)

        self.image_decoder = LensingDecoder(
            psf=survey.psf,
            bands=survey.BANDS,
            flux_calibration_dict=survey.flux_calibration_dict,
            ref_band=prior.reference_band,
        )
