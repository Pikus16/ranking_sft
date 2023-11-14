from typing import Dict, List, Optional
import json
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

QUERY_STR = 'Query: '

def process_gpt_input(input_str: str) -> Dict:
    assert input_str.startswith(QUERY_STR)
    if '["Passage 1' in input_str:
        passage_1_name = '["Passage 1'
    elif "['Passage 1" in input_str:
        passage_1_name = "['Passage 1"
    else:
        raise ValueError()
    passage_start_ind = input_str.index(passage_1_name)
    query = input_str[len(QUERY_STR):passage_start_ind].strip()
    passages = eval(input_str[passage_start_ind:])
    #assert len(passages) == 5
    enum_passages = {}
    for i, p in enumerate(passages):
        assert p.startswith(f"Passage {i+1}:")
        enum_passages[i+1] = p[len(f'Passage {i+1}:'):].strip()
    return {
        "query" : query,
        **enum_passages
    }

def process_gpt_output(output_str: str) -> Dict:
    # TODO: add reasoning output
    ranking = output_str.split('.')[-1].split('"')[-1].strip()
    assert ranking.startswith('Ranking:')
    ranking = ranking[len('Ranking:'):].strip()
    ranking = [int(x) for x in ranking.split(',')]
    #assert sorted(ranking) == list(range(1,6))
    return {
        'ranking': ranking
    }

def process_gpt_ranking(ranking: Dict) -> Dict:
    assert sorted(ranking.keys()) == ['input', 'instruction', 'output']
    input_dict = process_gpt_input(ranking['input'])
    ranking_output = process_gpt_output(ranking['output'])
    #assert len(input_dict)-1 == len(ranking_output['ranking'])
    ranked_responses = {i: input_dict[x] for i,x in enumerate(ranking_output['ranking'][:5])}
    return {
        'instruction': ranking['instruction'].strip(),
        'query' : input_dict['query'],
        **ranked_responses
        #**input_dict,
        #'output': ranking_output,
    }

def read_gpt_ranking(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r') as f:
        gpt_output = f.readlines()
    gpt_output = [json.loads(x) for x in gpt_output]
    outputs = []
    for x in gpt_output:
        try:
            outputs.append(process_gpt_ranking(x))
        except:
            pass
    return pd.DataFrame(outputs)

def get_perc_first_correct(y_true: List[List[int]], y_pred: List[List[int]]):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert y_true.shape[1] == 5
    assert y_pred.shape[1] == 5
    assert len(y_true.shape) == 2
    assert len(y_pred.shape) == 2
    return np.mean(y_pred[:,0] == y_true[:,0])

def calc_ncdg(rankings, y_true) -> float:
    mod_rankings = []
    for yt in y_true:
        weights = [0] * 5
        for j,i in enumerate(yt):
            weights[i-1] = 5-j
        mod_rankings.append(weights)
    return ndcg_score(mod_rankings, rankings)