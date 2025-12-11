import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'

from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,padding='max_length', max_length=256)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("data")

tokenized_data = dataset.map(preprocess_function, batched=True, batch_size=1000, num_proc=4)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length")
accuracy = evaluate.load("accuracy")

id2label = {0: "1 звезда", 1: "2 звезды", 2: "3 звезды", 3: "4 звезды", 4: "5 звёзд"}
label2id = {"1 звезда": 0, "2 звезды": 1, "3 звезды": 2, "4 звезды": 3, "5 звёзд": 4}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=5, id2label=id2label, label2id=label2id
)
training_args = TrainingArguments(
    output_dir="yelp_review_classifier",
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=True,
    optim="adamw_torch",
    save_total_limit=2,
    dataloader_num_workers=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()