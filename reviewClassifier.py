from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
test_dataset = load_dataset("Yelp/yelp_review_full", data_files="test.csv")
train_dataset = load_dataset("Yelp/yelp_review_full", data_files="train.csv")

tokenized_test = test_dataset.map(preprocess_function, batched=True)
tokenized_train = train_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

id2label = {1: "1 звезда", 2: "2 звезды", 3: "3 звезды", 4: "4 звезды", 5: "5 звёзд"}
label2id = {"1 звезда": 1, "2 звезды": 2, "3 звезды": 3, "4 звезды": 4, "5 звёзд": 5}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=5, id2label=id2label, label2id=label2id
)