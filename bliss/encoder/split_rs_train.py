# Author: Qiaozhi Huang
# split dataset to training and validation set as pickle for faster pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def split_as_train_val(df, test_size, random_state):
    """
    df: DataFrame
    test_size: float
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    dataset_name = 'desc_dc2_run2.2i_dr6_truth_nona'
    dir = '/home/qiaozhih/bliss/data/redshift/dc2'
    path = os.path.join(dir, f'{dataset_name}.pkl')
    print('reading dataset')
    df = pd.read_pickle(path)
    print('spliting dataset')
    train_set, val_set = split_as_train_val(df, test_size=0.2, random_state=42)
    train_set.reset_index(drop=True, inplace=True)
    val_set.reset_index(drop=True, inplace=True)

    print('saving training and validation dataset')
    train_path = os.path.join(dir, f'{dataset_name}_train.pkl')
    val_path = os.path.join(dir, f'{dataset_name}_val.pkl')
    train_set.to_pickle(train_path)
    val_set.to_pickle(val_path)