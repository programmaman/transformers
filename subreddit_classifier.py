import json
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from torch import cuda

import scipy

# CUDA if on GPU
device = 'cuda' if cuda.is_available() else 'cpu'

# Instantiate Pre-Trained Model

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4, dropout=0.12,
                                                           attention_dropout=0.12)
trainer = None
labels_list = []


# Json To Pandas Function
def json_to_pandas(path):
    dataset = []
    with open(path) as F:
        for line in F.readlines():
            line = json.loads(line)
            newline = dict()
            newline['labels'] = 0 if line['subreddit'] == "Nerf" \
                else 1 if line['subreddit'] == "ukulele" \
                else 2 if line['subreddit'] == "newToTheNavy" \
                else 3
            # newline['labels'] = line['subreddit']
            newline['text'] = line['body']  # + '[SEP]' + line['comment']
            dataset.append(newline)
    dataset = pd.DataFrame.from_dict(dataset)
    return Dataset.from_pandas(dataset)


# Use Tokenizer
def tokenize_function(dataset):
    return tokenizer(dataset["text"], padding="max_length", truncation=True)


# Metric to use is Accuracy
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    global labels_list
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print("Labels:", labels)
    print("Predictions:", predictions)
    result = metric.compute(predictions=predictions, references=labels)
    labels_list = predictions
    return result


training_args = TrainingArguments(
    output_dir=r"C:\Users\Alec\OneDrive\Documents\Natural Language Processing\programming assignments\hw4\model_out",
    evaluation_strategy="steps",
    save_strategy="steps",
    num_train_epochs=12,
    max_steps=25000,
    save_steps=1000,
    eval_steps=1000,
    optim="adamw_torch",
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)


def classifySubreddit_train(trainFile):
    global model
    global trainer

    dataset_train = json_to_pandas(trainFile)
    train_test = dataset_train.train_test_split(train_size=0.9)
    tokenized_dataset_train = train_test["train"].map(tokenize_function, batched=True)
    tokenized_dataset_train = tokenized_dataset_train.remove_columns(["text"])

    tokenized_dataset_test = train_test["test"].map(tokenize_function, batched=True)
    tokenized_dataset_test = tokenized_dataset_test.remove_columns(["text"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    model.cuda()
    trainer.train()


def classifySubreddit_test(testFile):
    global model
    global trainer
    string_list = []
    dataset_test = json_to_pandas(testFile)
    tokenized_dataset_test = dataset_test.map(tokenize_function, batched=True)
    trainer.predict(tokenized_dataset_test)
    for i in range(len(labels_list)):
        if labels_list[i] == 0:
            string_list.append("Nerf")
        elif labels_list[i] == 1:
            string_list.append("ukelele")
        elif labels_list[i] == 2:
            string_list.append("newToTheNavy")
        elif labels_list[i] == 3:
            string_list.append("MarioMaker")
    return string_list
