# pylint: skip-file
# flake8: noqa
from os import environ
from pathlib import Path

import GCRCatalogs
import pandas as pd
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from bliss.catalog import SourceType
from bliss.surveys.dc2 import DC2, from_wcs_header_str_to_wcs


def get_lsst_params(
    lsst_catalog_tensors_dict, cur_image_wcs, image_lim, image_height_index, image_width_index
):
    lsst_ra = lsst_catalog_tensors_dict["ra"]
    lsst_dec = lsst_catalog_tensors_dict["dec"]
    lsst_pt, lsst_pr = cur_image_wcs.all_world2pix(lsst_ra, lsst_dec, 0)
    lsst_pt = torch.from_numpy(lsst_pt)
    lsst_pr = torch.from_numpy(lsst_pr)

    lsst_plocs = torch.stack((lsst_pr, lsst_pt), dim=-1)
    lsst_source_type = lsst_catalog_tensors_dict["truth_type"]
    lsst_flux = lsst_catalog_tensors_dict["flux"]

    x0_mask = (lsst_plocs[:, 0] > image_height_index * image_lim) & (
        lsst_plocs[:, 0] < (image_height_index + 1) * image_lim
    )
    x1_mask = (lsst_plocs[:, 1] > image_width_index * image_lim) & (
        lsst_plocs[:, 1] < (image_width_index + 1) * image_lim
    )
    lsst_x_mask = x0_mask * x1_mask
    # filter r band
    lsst_flux_mask = lsst_flux[:, 2] > 0
    # filter supernova
    lsst_source_mask = (lsst_source_type != 3).squeeze(1)
    lsst_mask = lsst_x_mask * lsst_flux_mask * lsst_source_mask

    lsst_plocs = lsst_plocs[lsst_mask, :] % image_lim
    lsst_source_type = torch.where(
        lsst_source_type[lsst_mask] == 2, SourceType.STAR, SourceType.GALAXY
    )
    lsst_flux = lsst_flux[lsst_mask, :]

    return lsst_plocs, lsst_source_type, lsst_flux


def generate_lsst_fullcat_files(
    split_results_path, split_ids, lsst_catalog_tensors_dict, output_path
):
    output_path.mkdir(parents=True, exist_ok=True)
    for split_id in split_ids:
        with open(split_results_path / split_id, "rb") as split_result_file:
            split_result = torch.load(split_result_file)
        cur_image_wcs = from_wcs_header_str_to_wcs(split_result["wcs_header_str"])
        image_lim = split_result["images"].shape[1]
        assert (
            split_result["images"].shape[1] == split_result["images"].shape[2]
        ), "image width should be equal to image height"

        cur_plocs, cur_source_type, cur_flux = get_lsst_params(
            lsst_catalog_tensors_dict,
            cur_image_wcs,
            image_lim,
            image_height_index=split_result["image_height_index"],
            image_width_index=split_result["image_width_index"],
        )

        with open(output_path / ("lsst_" + split_id), "wb") as output_file:
            torch.save(
                {
                    "plocs": cur_plocs.clone(),
                    "source_type": cur_source_type.clone(),
                    "flux": cur_flux.clone(),
                },
                output_file,
            )


if __name__ == "__main__":
    print("+" * 100, flush=True)
    print("initialization begins", flush=True)
    environ["BLISS_HOME"] = str(Path().resolve().parents[1])

    output_dir = Path("./DC2_bootstrap_output/")
    output_dir.mkdir(parents=True, exist_ok=True)

    with initialize(config_path=".", version_base=None):
        notebook_cfg = compose("notebook_config")
    print("initialization ends", flush=True)
    print("+" * 100, flush=True)

    print("+" * 100, flush=True)
    print("load dc2", flush=True)
    dc2: DC2 = instantiate(notebook_cfg.surveys.dc2)
    dc2.prepare_data()
    dc2.setup()
    dc2_test_dataset = dc2.test_dataset
    print("+" * 100, flush=True)

    print("+" * 100, flush=True)
    print("load lsst catalog", flush=True)
    GCRCatalogs.set_root_dir("/data/dc2/")
    lsst_catalog_gcr = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_object_with_truth_match")
    lsst_catalog_sub = lsst_catalog_gcr.get_quantities(
        [
            "id_truth",
            "objectId",
            "ra",
            "dec",
            "truth_type",
            "cModelFlux_u",
            "cModelFluxErr_u",
            "cModelFlux_g",
            "cModelFluxErr_g",
            "cModelFlux_r",
            "cModelFluxErr_r",
            "cModelFlux_i",
            "cModelFluxErr_i",
            "cModelFlux_z",
            "cModelFluxErr_z",
            "cModelFlux_y",
            "cModelFluxErr_y",
        ]
    )
    lsst_catalog_df = pd.DataFrame(lsst_catalog_sub)
    lsst_catalog_tensors_dict = {
        "truth_type": torch.tensor(lsst_catalog_df["truth_type"].values).view(-1, 1),
        "flux": torch.cat(
            [
                torch.tensor(flux.values).view(-1, 1)
                for flux in [
                    lsst_catalog_df["cModelFlux_g"],
                    lsst_catalog_df["cModelFlux_i"],
                    lsst_catalog_df["cModelFlux_r"],
                    lsst_catalog_df["cModelFlux_u"],
                    lsst_catalog_df["cModelFlux_y"],
                    lsst_catalog_df["cModelFlux_z"],
                ]
            ],
            dim=1,
        ),
        "ra": torch.tensor(lsst_catalog_df["ra"].values),
        "dec": torch.tensor(lsst_catalog_df["dec"].values),
    }
    print("+" * 100, flush=True)

    print("+" * 100, flush=True)
    print("generate lsst split results", flush=True)
    split_results_path = Path(
        "/data/scratch/dc2local/run2.2i-dr6-v4/coadd-t3828-t3829/deepCoadd-results/split_results/"
    )
    split_ids = [split_file_path.name for split_file_path in dc2_test_dataset.split_files_list]
    print(f"there are {len(split_ids)} split_ids", flush=True)
    lsst_split_result_dir = output_dir / "lsst_split_results"
    lsst_split_result_dir.mkdir(exist_ok=True)

    generate_lsst_fullcat_files(
        split_results_path, split_ids, lsst_catalog_tensors_dict, lsst_split_result_dir
    )

    # base_param_dict = {"split_results_path": split_results_path,
    #                    "lsst_catalog_tensors_dict": lsst_catalog_tensors_dict,
    #                    "output_path": lsst_split_result_dir}
    # processes_num = 4
    # split_ids_chunks = [split_ids[i:(i + len(split_ids) // processes_num)] for i in range(0, len(split_ids), len(split_ids) // processes_num)]
    # with multiprocessing.Pool(processes=processes_num) as process_pool:
    #             process_pool.starmap(
    #                 generate_lsst_fullcat_files,
    #                 [[split_results_path, split_ids_chunk, lsst_catalog_tensors_dict, lsst_split_result_dir] for split_ids_chunk in split_ids_chunks],
    #                 chunksize=4,
    #             )
    print("generation ends", flush=True)
    print("+" * 100, flush=True)
