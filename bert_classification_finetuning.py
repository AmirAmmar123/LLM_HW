import sys
import os
import numpy as np

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score



DEBUG = True               
DEBUG_MODEL_DIR = "./debug_saved_model"


def main():

    if len(sys.argv) < 2:
        print("Usage: python bert_classification_finetuning.py <path/to/imdb_subset>")
        return

    subset_path = sys.argv[1]


    if os.path.exists(subset_path):
        if DEBUG:
            print(f"Loading dataset from disk: {subset_path}")
        subset = load_from_disk(subset_path)
    else:
        if DEBUG:
            print("Dataset not found. Downloading IMDB and creating subset...")
        dataset = load_dataset("imdb")
        subset = dataset["train"].shuffle(seed=42).select(range(500))
        subset.save_to_disk(subset_path)

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    subset = subset.rename_column("label", "labels")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    tokenized_dataset = subset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )


    split_dataset = tokenized_dataset.train_test_split(
        test_size=0.2,
        seed=42
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    if DEBUG and os.path.exists(DEBUG_MODEL_DIR):
        print("Loading model from DEBUG directory...")
        model = AutoModelForSequenceClassification.from_pretrained(DEBUG_MODEL_DIR)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )


    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}


    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",      
        save_strategy="epoch",      
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        report_to="none",
    )



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    if DEBUG:
        print("Starting training...")
    trainer.train()


    if DEBUG:
        os.makedirs(DEBUG_MODEL_DIR, exist_ok=True)
        trainer.save_model(DEBUG_MODEL_DIR)
        tokenizer.save_pretrained(DEBUG_MODEL_DIR)
        print("Model saved (DEBUG mode).")


    if DEBUG:
        print("Evaluating on test set...")
    metrics = trainer.evaluate()

    accuracy = metrics["eval_accuracy"]
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
