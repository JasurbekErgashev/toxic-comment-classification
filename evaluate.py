import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from clean_text import clean_text
from dataset_class import ToxicDataset


def evaluate(text):
    data = {"id": [0], "comment_text": [text]}
    df = pd.DataFrame(data)
    df["comment_text"] = df["comment_text"].apply(clean_text)

    model = torch.load("dsbert_toxic_balanced.pt", map_location=torch.device("cpu"))
    model.eval()

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_output_set = ToxicDataset(df, tokenizer, 128, eval_mode=True)
    test_output_loader = DataLoader(test_output_set, batch_size=16, shuffle=False)

    test_predictions = []
    with torch.no_grad():
        for batch in test_output_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().detach().numpy())

    test_ids = df["id"]
    predictions_df = pd.DataFrame({"id": test_ids, "label": test_predictions})
    return str(predictions_df.iloc[0]["label"])
