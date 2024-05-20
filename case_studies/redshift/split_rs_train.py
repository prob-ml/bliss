# Author: Qiaozhi Huang
# split dataset to training and validation set as pickle for faster pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import click

def split_as_train_val(df, test_size, random_state):
    """
    df: DataFrame
    test_size: float
    """
    train_val_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
    train_set, val_set = train_test_split(train_val_set, test_size=test_size, random_state=random_state)
    return train_set, val_set, test_set

@click.command()
@click.option('--source', type=str)
@click.option('--outname', type=str)
def main(source, outname):
    path = source
    print('reading dataset')
    df = pd.read_pickle(path)
    print('spliting dataset')
    train_set, val_set, test_set = split_as_train_val(df, test_size=0.2, random_state=42)
    train_set.reset_index(drop=True, inplace=True)
    val_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)

    print('saving training and validation dataset')
    train_path = f'{outname}_train.pkl'
    val_path = f'{outname}_val.pkl'
    test_path = f'{outname}_test.pkl'
    train_set.to_pickle(train_path)
    val_set.to_pickle(val_path)
    test_set.to_pickle(test_path)
    

if __name__ == "__main__":
    main()