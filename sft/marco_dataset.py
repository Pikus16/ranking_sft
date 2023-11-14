import torch
import pandas as pd
from train_utils import format_prompt_finetuning, seed_all
from trl import DataCollatorForCompletionOnlyLM

class MarcoDataset(torch.utils.data.Dataset):

    def __init__(self, df: pd.DataFrame, seed: int = 42):
        self.df = df #pd.read_csv(csv_path)
        seed_all(seed)
        self.preprocess()
        super().__init__()

    def preprocess(self):
        self.texts, self.labels = [], []
        for _, row in self.df.iterrows():
            prompt = format_prompt_finetuning(row['query'], row[['0','1','2','3','4']].to_list())
            self.texts.append(prompt)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]
        
    def to_csv(self, csv_path):
        pd.DataFrame({'text' : self.texts}).to_csv(csv_path, index=False)