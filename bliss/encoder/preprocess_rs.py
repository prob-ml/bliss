# Author: Qiaozhi Huang
# Aim to preprocess the DC2 data for redshift prediction

import os
import math
import numpy as np
import torch
import pandas as pd
import GCRCatalogs
from GCR import GCRQuery
from tqdm import tqdm

def load_dataset(rootdir='/nfs/turbo/lsa-regier/', dataset='desc_dc2_run2.2i_dr6_truth'):
    """
    load datadset
    ----
    Params:
    rootdir: str
    dataset: str, name of the datset
    ----
    Return: 
    dataset
    """
    # need to do this in accordance with instructions at https://data.lsstdesc.org/doc/install_gcr
    GCRCatalogs.set_root_dir(rootdir)
    GCRCatalogs.get_root_dir()
    truth_cat = GCRCatalogs.load_catalog('desc_dc2_run2.2i_dr6_truth')
    return truth_cat

def load_quantities(dataset, quantities):
    """
    load quantities in Dataframe
    ----
    Params:
    dataset: GCR returned dataset
    quantities: list, variables want to load
    ----
    Return:
    Dataframe
    """
    all_truth_data = {}
    all_quantities = ["flux_u", "flux_g", "flux_r", "flux_i", "flux_z", "flux_y",
                "mag_u", "mag_g", "mag_r", "mag_i", "mag_z", "mag_y",
                "truth_type", "redshift",
                "id", "match_objectId", "cosmodc2_id", "id_string"]
    for q in tqdm(quantities):
        assert q in all_quantities
        this_field = dataset.get_quantities([q])
        all_truth_data[q] = this_field[q]
        print('Finished {}'.format(q))
    return pd.DataFrame(all_truth_data)

def save_pickle(dataframe, path):
    """
    save in pickle format
    ----
    Params:
    dataframe: Dataframe
    path: str
    ----
    Returns:
    None
    """
    dataframe.to_pickle(path)

if __name__== "__main__":
    rootdir = '/nfs/turbo/lsa-regier/'
    dataset_name = 'desc_dc2_run2.2i_dr6_truth'
    quantities = ["mag_u", "mag_g", "mag_r", "mag_i", "mag_z", "mag_y", "redshift"]
    path = f'/home/qiaozhih/bliss/data/redshift/dc2/{dataset_name}.pkl'

    dataset = load_dataset(rootdir, dataset_name)
    dataset = load_quantities(dataset, quantities)
    save_pickle(dataset, path)


