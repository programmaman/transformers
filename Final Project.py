# Alec Braynen, Logan Fields, and Ben Kreiger
# Natural Language Processing
# Final Project

# Paper replicated: Truth of Varying Shades: Analyzing Language in Fake News and Political Fact-Checking

from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          Trainer,
                          TrainingArguments)
from datasets import (Dataset, load_dataset, load_metric)
import pandas as pd
import numpy as np
import evaluate
from scipy.special import softmax

# This is designed based on the multiclassification task, not the binary task

# load dataset
dataset = load_dataset("csv", delimiter='\t', data_files={"train": 'train.csv',
                                                          "dev": 'dev.csv',
                                                          "test": 'test.csv'})

# rename ratings column to labels
dataset = dataset.rename_column("Rating", "labels")

# Visualize the distribution for the training set
print("Label 0:", dataset['train']['labels'].count(0))
print("Label 1:", dataset['train']['labels'].count(1))
print("Label 2:", dataset['train']['labels'].count(2))
print("Label 3:", dataset['train']['labels'].count(3))
print("Label 4:", dataset['train']['labels'].count(4))
print("Label 5:", dataset['train']['labels'].count(5))
# Visualize the dataset
dataset

# instantiate model distilbet
model_name = "distilbert-base-uncased" # Started with default distilbert
metric = "f1" # Change this to test different metrics
num_labels = 6 # 2 or 6 depending on task
epochs = 10 # Starts overfitting if higher, set to 10 for efficiency
learning_rate = 5e-6 # Tested 5e-4, 1e-4, 5e-5, 1e-5, and 5e-6
batch_size = 16 # Tested 8 and 16

# Tested 0, 0.05, 0.1, 0.15, 0.2, 0.25, and 0.3 for dropout
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, dropout=0.05)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tested with including the speaker for each statement and using linear regression, but neither showed significant difference.
# Tested using Bart Summarization as training data and we got an F1 score of 0.18, future work could be different variations of bart + politifact data

# tokenize dataset
def tokenize_dataset(dataset):
  dataset = dataset.map(lambda examples : tokenizer(examples["Statement"], padding=True, truncation=True), batched=True)
  return dataset

# We tested macro-F1, as that was the metric the paper used, but we also tested other metrics for choosing the best model 
def compute_metrics(eval_pred):
  if metric == "roc_auc": # Good for showing the model's ability to distinguish between classes, but will not perform better than F1 with unbalanced classes
    evaluation = evaluate.load("roc_auc", "multiclass")
    logits, labels = eval_pred
    # OVR lets us treat this as a series of binary tasks
    results = evaluation.compute(prediction_scores=softmax(logits, axis=-1), references=labels, multi_class="ovr", average="macro")
  elif metric == "matthews_correlation": # Uses TP, FP, TN, and FN to determine if each class is predicted well, regardless of imbalance
    evaluation = load_metric("matthews_correlation")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    results = evaluation.compute(predictions=preds, references=labels)
  elif metric == "f1": # We tested micro- and macro-F1; micro-F1 produced better results (0.237), but was likely skewed due to class imbalance
    evaluation = load_metric("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    results = evaluation.compute(predictions=preds, references=labels, average="macro")
  elif metric == "accuracy": # Accuracy can show us how our model performs compared to random with this many classes
    evaluation = load_metric("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    results = evaluation.compute(predictions=preds, references=labels)
  return results

# training arguments for model
training_args = TrainingArguments(
    output_dir = "./saves",
    evaluation_strategy="epoch",
    num_train_epochs=epochs,
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0, # Tested 0, 1e-5, 1e-4, 1e-3, and 1e-2
    metric_for_best_model="f1", # We output this at the end, after hyperparameter tuning, to show how our model compares to the paper
    load_best_model_at_end=True
)

dataset = tokenize_dataset(dataset)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Use best model to predict on test set
trainer.predict(dataset["test"])
