import numpy as np 
import pandas as pd
import ast
import torch
from datasets import load_dataset
import evaluate
from sklearn.model_selection import train_test_split
from transformers import(
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorForTokenClassification

)
from utilities_ner_functions import(
    tokenize_and_align_labels_black_box,
    baseDataset

)

import warnings
warnings.filterwarnings("ignore")
# Set up logging to both console and file

#GENERAL VARIABLES
CSV_PATH = r'/home/tensorboard/Documentos/1. D4R/3. NER/training/dataset_NER_Pineda_completo_06-06-24.csv'
MODEL_CHECKPOINT = "dccuchile/bert-base-spanish-wwm-cased"
MODEL_SUMMARY_NAME = 'bert-base-spanish-wwm-cased'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIRECTORY = "/home/tensorboard/Documentos/1. D4R/8. Models/bert-base-spanish-wwm-cased-Linear-NER-Religious-Pineda"
LEARNING_RATE = 2e-5 # hemos provado 2e-5 (overfit), 2e-6 (más epochs por tener noisy data) posibilidades 5e-5, 6e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 10
WEIGHT_DECAY = 0.05
EVALUATION_STRATEGY = "steps"
EVAL_STEPS = 100
SAVE_STRATEGY = "steps"
SAVE_STEPS = 100
LOGGING_STEPS = 10
#GRADIENT_ACCUMULATION_STEPS = 4 #For RoBERTa Large
SAVE_ONLY_MODEL = True
OUTPUT_DIRECTORY = f"/home/tensorboard/Documentos/1. D4R/8. Models/Full-Linear-NER-{str(MODEL_SUMMARY_NAME)}-{str(LEARNING_RATE)}-lr-{str(NUM_TRAIN_EPOCHS)}-epochs"
LR_SCHEDULER_TYPE = 'linear' # podemos probar a cambiar a constat rate
#['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr']


df = pd.read_csv(CSV_PATH)
df = df.drop('Unnamed: 0', axis=1)
df['sentence'] = df['sentence'].apply(ast.literal_eval)
df['tags'] = df['tags'].apply(ast.literal_eval)
df['encoded_tags'] = df['encoded_tags'].apply(ast.literal_eval)


labels_to_ids = {'O':0,'GOD': 1, 'JUS': 2, 'CHRI': 3, 'SACRA': 4, 'PER': 5, 'PLACE': 6}


ids_to_labels = {v:k for k, v in labels_to_ids.items()}

labels_list = [x for x in ids_to_labels.values()]

number_of_labels = len(labels_list)
df_prepared = df[['sentence', 'encoded_tags']]

train_df, eval_df = train_test_split(df_prepared, train_size= 0.9, random_state=42, shuffle=True)


tokenizer= AutoTokenizer.from_pretrained(MODEL_CHECKPOINT,  add_prefix_space=True)

tokenized_inputs_train = tokenize_and_align_labels_black_box(
                                                                train_df,
                                                                tokenizer,
                                                                token_column='sentence',
                                                                tag_column='encoded_tags'
                                                            )

tokenized_inputs_eval = tokenize_and_align_labels_black_box(
                                                                eval_df,
                                                                tokenizer,
                                                                token_column='sentence',
                                                                tag_column='encoded_tags'
                                                            )


model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=number_of_labels, id2label=ids_to_labels, label2id=labels_to_ids
)


seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    """
    Computes the precision, recall, F1, and accuracy scores for the given predictions and labels.

    Args:
        p (tuple): A tuple containing the predictions and labels.

    Returns:
        dict: A dictionary containing the precision, recall, F1, and accuracy scores.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [labels_list[p] for (p, l) in zip(prediction, label) if l!= -100] # type: ignore
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels_list[l] for (p, l) in zip(prediction, label) if l!= -100] # type: ignore
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"], # type: ignore
        "recall": results["overall_recall"], # type: ignore
        "f1": results["overall_f1"], # type: ignore
        "accuracy": results["overall_accuracy"], # type: ignore
    }

# Making the data collator and the datasets for train and eval
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

dataset_train= baseDataset(tokenized_inputs_train)
dataset_eval =baseDataset(tokenized_inputs_eval)

# Training Process
training_args = TrainingArguments(
    output_dir=OUTPUT_DIRECTORY,
    learning_rate=LEARNING_RATE ,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    evaluation_strategy=EVALUATION_STRATEGY,
    eval_steps=EVAL_STEPS,
    save_strategy=SAVE_STRATEGY,
    save_steps = SAVE_STEPS,
    save_only_model= SAVE_ONLY_MODEL,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    logging_steps= LOGGING_STEPS


) # type: ignore

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)



trainer.train()
trainer.save_model(output_dir=OUTPUT_DIRECTORY)