from torch.utils.data import Dataset
import pandas as pd


class SentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.text = self.df.text
        self.label_id = self.df.label_id

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.label_id[idx]
        tokenized_text = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].squeeze(0)
        attention_mask = tokenized_text["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }



