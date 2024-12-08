import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


data = pd.read_csv('/content/BERT.csv')

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

model_name = 'xlm-roberta-base'
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaForTokenClassification.from_pretrained(model_name, num_labels=len(set(" ".join(data['POS_Tags']).split())))

unique_tags = list(set(" ".join(data['POS_Tags']).split()))
tag2id = {tag: idx for idx, tag in enumerate(unique_tags)}
id2tag = {idx: tag for tag, idx in tag2id.items()}

class POSDataset(Dataset):
    def __init__(self, data, tokenizer, tag2id, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['Sentence']
        tags = self.data.iloc[idx]['POS_Tags'].split()

        encoding = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        labels = [self.tag2id.get(tag, -100) for tag in tags]
        labels = labels + [-100] * (self.max_length - len(labels))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels)
        }

train_dataset = POSDataset(train_data, tokenizer, tag2id)
test_dataset = POSDataset(test_data, tokenizer, tag2id)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch"
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    mask = labels != -100

    labels = labels[mask]
    preds = preds[mask]

    return classification_report(labels, preds, target_names=list(tag2id.keys()), output_dict=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

predictions, labels, _ = trainer.predict(test_dataset)
preds_flat = np.argmax(predictions, axis=2)

pred_tags = [
    [id2tag[label] for label, mask in zip(pred, labels[idx]) if mask != -100]
    for idx, pred in enumerate(preds_flat)
]
true_tags = [
    [id2tag[label] for label, mask in zip(true, labels[idx]) if mask != -100]
    for idx, true in enumerate(labels)
]

flat_pred_tags = [tag for sent in pred_tags for tag in sent]
flat_true_tags = [tag for sent in true_tags for tag in sent]

if len(flat_pred_tags) == len(flat_true_tags):
    test_data['Predicted_POS_Tags'] = [" ".join(tags) for tags in pred_tags]
    test_data.to_csv("RoBERTaoutput.csv", index=False)

    print("Model Evaluation:")
    print(classification_report(flat_true_tags, flat_pred_tags, target_names=list(tag2id.keys())))
else:
    print(f"Error: Number of predicted tags ({len(flat_pred_tags)}) does not match number of true tags ({len(flat_true_tags)})")
