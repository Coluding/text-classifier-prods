import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, Dataset, ClassLabel
import pandas as pd
import torch
from typing import Tuple


def setup_datasets(csv_data: str, minimum_num_instances: int = 80, seed: int = 123, split_size: float = 0.2,
                   delim: str = "{") -> (
        Tuple)[datasets.Dataset, datasets.Dataset, dict]:
    df = pd.read_csv(csv_data, delimiter=delim)

    if len(df.columns) == 3:
        df = df.iloc[:, 1:]
    df.columns = ['text', 'label']

    # Filter out labels with less than minimum_num_instances
    label_counts = df['label'].value_counts()
    labels_to_keep = label_counts[label_counts >= minimum_num_instances].index
    df_reduced = df[df['label'].isin(labels_to_keep)]
    label_mapping = {label: i for i, label in enumerate(df['label'].unique())}
    df_reduced['label'] = df_reduced['label'].map(label_mapping)

    dataset = Dataset.from_pandas(df_reduced)

    train_test_split = dataset.train_test_split(test_size=split_size, seed=seed)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    return train_dataset, test_dataset, label_mapping


class CustomCollator:
    def __init__(self, tokenizer, for_dataloader: bool = True,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.tokenizer = tokenizer
        self.for_dataloader = for_dataloader
        self.device = device

    def __call__(self, batch):
        if self.for_dataloader:
            texts = batch['text']
            labels = batch['label']
        else:
            texts = [item['text'] for item in batch]
            labels = [item['label'] for item in batch]
        encoding = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoding['labels'] = torch.tensor(labels)
        return encoding



