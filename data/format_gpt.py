import sys
sys.path.append('..')
import click
from utils import read_gpt_ranking
import pandas as pd

@click.command()
@click.option('--input', '-i', 'input_file', type=click.Path(exists=True), required=True)
@click.option('--output', '-o', 'output_file', type=click.Path(), required=True)
def main(input_file: str, output_file: str):
    df = read_gpt_ranking(input_file)
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()