import numpy as np 
import torch
from datasets import load_dataset
import evaluate
from transformers import(
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorForTokenClassification

)

import logging
import sys
import warnings
warnings.filterwarnings("ignore")
# Set up logging to both console and file
FILE_LOG_NAME = "roberta-large-bne-Linear-NER"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(f"{FILE_LOG_NAME}.log")])
#GENERAL VARIABLES

MODEL_CHECKPOINT = "PlanTL-GOB-ES/roberta-base-bne"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIRECTORY = "/home/tensorboard/Documentos/1. D4R/8. Models/roberta-large-bne-Linear-Religious-NER"
LEARNING_RATE = 5e-6
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 1
WEIGHT_DECAY = 0.01
EVALUATION_STRATEGY = "steps"
EVAL_STEPS = 10000
SAVE_STRATEGY = "steps"
SAVE_STEPS = 10000
DATASET_NAME = "hlhdatscience/es-ner-massive"
GRADIENT_ACCUMULATION_STEPS = 4 #For RoBERTa Large
SAVE_ONLY_MODEL = True
LR_SCHEDULER_TYPE = 'constant'
#['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr']

#Preparing Dataset
logging.info("preparing dataset....")
dataset = load_dataset(DATASET_NAME)
labels_to_ids = {
      "O": 0,
      "PER": 1,
      "ORG": 2,
      "LOC": 3,
      "MISC": 4
  }
ids_to_labels = {v:k for k, v in labels_to_ids.items()}
labels_list = list(ids_to_labels.values())
number_of_labels = len(ids_to_labels)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, add_prefix_space=True)
def tokenize_and_align_labels_HF_dataset(examples):
    tokenized_inputs = tokenizer(examples["Tokens"], truncation=True, is_split_into_words=True) 

    labels = []
    for i, label in enumerate(examples[f"Tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels_HF_dataset, batched=True)
logging.info("dataset prepared and tokenized")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

logging.info("Loading model...")

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=number_of_labels, id2label=ids_to_labels, label2id=labels_to_ids
)

logging.info("Model Loaded")
logging.info("preparing metrics...")
example = dataset["train"][0] # type: ignore
label_list = [i for i in ids_to_labels.values()]
labels = [label_list[i] for i in example[f"Tags"]]
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"], # type: ignore
        "recall": results["overall_recall"],# type: ignore
        "f1": results["overall_f1"], # type: ignore
        "accuracy": results["overall_accuracy"], # type: ignore
    }
logging.info("metrics prepared")

logging.info("Starting training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIRECTORY,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    evaluation_strategy=EVALUATION_STRATEGY,
    eval_steps=EVAL_STEPS,
    save_strategy=SAVE_STRATEGY,
    save_steps=SAVE_STEPS,
    #gradient_accumulation_steps= GRADIENT_ACCUMULATION_STEPS,
    save_only_model=SAVE_ONLY_MODEL

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"], # type: ignore
    eval_dataset=tokenized_dataset["test"], # type: ignore
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()

logging.info("Training finished")

