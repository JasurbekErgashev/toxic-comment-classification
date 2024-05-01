import gradio as gr
from evaluate import evaluate
import torch.nn as nn
from transformers import DistilBertModel


class DistilBERT_Model(nn.Module):
    def __init__(self, num_labels):
        super(DistilBERT_Model, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.distilbert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits


def greet(text):
    output = evaluate(text)
    if output == "0":
        return "Toxic ☢️"
    else:
        return "Non-toxic 🤘"


demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
