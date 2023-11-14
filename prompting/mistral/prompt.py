
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('../../sft')
sys.path.append('../../')
from utils import get_perc_first_correct, calc_ncdg
from train_utils import remove_ranking, process_ranking, get_labels
import click
import numpy as np


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    print('loading model')
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model.to(DEVICE)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_test_data():
    df = pd.read_csv('../../data/split/test.csv')
    return df

def get_response(prompt, model, tokenizer):
    # TODO - add batching
    model_inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(**model_inputs, max_new_tokens=20, do_sample=False)
    return tokenizer.batch_decode(generated_ids)[0]

def perform_inference(df, model, tokenizer):
    results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # TODO: add error handling 
        prompt, _ = remove_ranking(row.text)
        results.append(
            get_response(prompt, model, tokenizer) 
        )
    return results

def process_rankings(results):
    rankings = []
    for r in results:
        res = process_ranking(r)
        assert len(res) <= 5 and len(res) >= 3
        if len(res) < 5:
            missing = list(set(np.arange(1,6)) - set(res))
            res += missing
        assert len(res) == 5
        rankings.append(res)
    return rankings

@click.command()
@click.option('--output', '-o', 'output_file', type=click.Path(), required=True)
def main(output_file):
    model, tokenizer = load_model()
    df = load_test_data()
    results = perform_inference(df, model, tokenizer)
    df['raw_results'] = results
    df.to_csv(output_file, index=False)
    df['rankings'] = process_rankings(results)
    df['y_true'] = get_labels(df)
    perc_correct = get_perc_first_correct(df['y_true'].to_list(), df['rankings'].to_list())
    ncdg = calc_ncdg(df['y_true'].to_list(), df['rankings'].to_list())
    print(f'Perc 1st correct: {perc_correct}')
    print(f'NCDG: {ncdg}')
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()