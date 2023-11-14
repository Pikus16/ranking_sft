import sys
sys.path.append('../sft')
import click
from marco_dataset import MarcoDataset
from train_utils import seed_all
import pandas as pd
import numpy as np
from typing import Dict
import os

def get_dataset(csv_path: str, train_size: float = 0.6, val_size:float = 0.2, test_size: float = 0.2, seed: int = 42) -> Dict: 
    seed_all(seed)
    assert train_size + test_size + val_size == 1.0
    
    val_size = train_size + val_size
    df = pd.read_csv(csv_path)
    train_df, validate_df, test_df = np.split(df.sample(frac=1, random_state=seed), 
                       [int(train_size*len(df)), int(val_size*len(df))])
    train_dataset = MarcoDataset(train_df)
    val_dataset = MarcoDataset(validate_df)
    test_dataset = MarcoDataset(test_df)
    return {
        "train" :train_dataset, "val" :val_dataset, "test" : test_dataset
    }

@click.command()
@click.option('--input', '-i', 'input_file', type=click.Path(exists=True), required=True)
@click.option('--output', '-o', 'output_dir', type=click.Path(), required=True)
def main(input_file: str, output_dir: str):
    split_ds = get_dataset(input_file)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for name, ds in split_ds.items():
        ds.to_csv(os.path.join(output_dir, f"{name}.csv"))

if __name__ == '__main__':
    main()