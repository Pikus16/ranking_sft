from typing import List
import numpy as np
import random
import torch
import os

INSTR_NAIVE = """
Given a query and five passages, rank the passages based on how relevant they are to the query. Output only the ranking of the passages in order of relevance, separated by commas.
""".strip()

RESPONSE_TEMPLATE = "## Ranking:"
RESPONSE_TEMPLATE_TOKENS = [ 1064, 15938,   288, 28747, 28705]

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

def format_prompt_finetuning(query: str, passages: List[str], instr: str = INSTR_NAIVE, shuffle: bool = True) -> str:
    
    prompt = f"{INSTR_NAIVE}\n\nQuery: {query}\n"
    inds = np.arange(5)
    if shuffle:
        np.random.shuffle(inds)
    for j, i in enumerate(inds):
        prompt += f"Passage {j+1}: {passages[i]}\n"
    prompt += f'\n{RESPONSE_TEMPLATE}'
    for i in inds:
        prompt += f" {i+1},"
    prompt = prompt[:-1]
    return prompt