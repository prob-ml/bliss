# Author: Qiaozhi Huang
# Aim to preprocess the DC2 data for redshift prediction


import click
import GCRCatalogs
import pandas as pd
from tqdm import tqdm


def load_dataset(
    rootdir: str = "/nfs/turbo/lsa-regier/", dataset: str = "desc_dc2_run2.2i_dr6_truth"
) -> None:
    """Load datadset.

    Args:
        rootdir: str
        dataset: str, name of the datset

    Returns:
        dataset
    """
    # need to do this in accordance with instructions at https://data.lsstdesc.org/doc/install_gcr
    GCRCatalogs.set_root_dir(rootdir)
    GCRCatalogs.get_root_dir()
    return GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_truth")


def load_quantities(dataset, quantities: list) -> pd.DataFrame:
    """Load quantities in Dataframe.

    Args:
        dataset: GCR returned dataset
        quantities: list, variables want to load

    Returns:
        Dataframe
    """
    all_truth_data = {}
    all_quantities = [
        "flux_u",
        "flux_g",
        "flux_r",
        "flux_i",
        "flux_z",
        "flux_y",
        "mag_u",
        "mag_g",
        "mag_r",
        "mag_i",
        "mag_z",
        "mag_y",
        "truth_type",
        "redshift",
        "id",
        "match_objectId",
        "cosmodc2_id",
        "id_string",
    ]
    for q in tqdm(quantities):
        assert q in all_quantities
        this_field = dataset.get_quantities([q])
        all_truth_data[q] = this_field[q]
        print(f"Finished {q}")  # noqa: WPS421
    return pd.DataFrame(all_truth_data)


def save_pickle(dataframe: pd.DataFrame, path: str) -> None:
    """Save in pickle format.

    Args:
        dataframe: Dataframe
        path: str
    """
    dataframe.to_pickle(path)


def get_rid_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Get rid of nan value.

    Args:
        df: Dataframe

    Returns:
        Dataframe
    """
    return df.dropna()


@click.command()
@click.option("--out", help="output_path", type=str)
def main(out):
    rootdir = "/nfs/turbo/lsa-regier/"
    dataset_name = "desc_dc2_run2.2i_dr6_truth"
    quantities = ["mag_u", "mag_g", "mag_r", "mag_i", "mag_z", "mag_y", "redshift"]
    path = out  # PATH TO SAVE

    dataset = load_dataset(rootdir, dataset_name)
    dataset = load_quantities(dataset, quantities)
    dataset = get_rid_nan(dataset)
    save_pickle(dataset, path)


if __name__ == "__main__":
    main()
