import torch
from torch.utils.data import Dataset


class ToxicDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, eval_mode: bool = False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval_mode = eval_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["comment_text"]
        if not self.eval_mode:
            label = self.data.iloc[idx]["label"]
        else:
            label = 0

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }
