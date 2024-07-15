import ast
import numpy as np
import torch # type: ignore
from torch import nn
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split
import evaluate # type: ignore
from transformers import( # type: ignore
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)
from TokenClassificationEncoderModels import RobertaTokenClassifier_With_LSTM, BertTokenClassifier_With_LSTM

from utilities_ner_functions import(
    tokenize_and_align_labels_black_box,
    baseDataset,
    compute_metrics,
)
from torchinfo import summary
import warnings
warnings.filterwarnings("ignore")

#GENERAL variables

CSV_PATH = r'/home/tensorboard/Documentos/1. D4R/3. NER/training/dataset_NER_completo_28-05-24.csv'
MODEL_NER_CHECKPOINTS = "/media/tensorboard/PRODUCCIÓN Y ACADÉMICO/TRABAJO/D4R_models/NER_models/General_models/bert-base-spanish-wwm-cased-Linear-NER/checkpoint-50000"
MODEL_ORIGINAL_CHECKPOINT = 'dccuchile/bert-base-spanish-wwm-cased'
MODEL_SUMMARY_NAME = 'bert-base-spanish-wwm-cased'
SUMMARY_BATCHSIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 10
PER_DEVICE_EVAL_BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 50
WEIGHT_DECAY = 0.01
EVALUATION_STRATEGY = "steps"
EVAL_STEPS = 100
LOGGING_STEPS = 100
SAVE_STRATEGY = "steps"
SAVE_STEPS = 500
SAVE_ONLY_MODEL = True
OUTPUT_DIRECTORY = f"/home/tensorboard/Documentos/1. D4R/8. Models/TF-LSTM-{str(MODEL_SUMMARY_NAME)}-{str(LEARNING_RATE)}-lr-{str(NUM_TRAIN_EPOCHS)}-epochs"
LR_SCHEDULER_TYPE = 'constant'
#['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr']
df = pd.read_csv(CSV_PATH)
df = df.drop('Unnamed: 0', axis=1)
df['sentence'] = df['sentence'].apply(ast.literal_eval)
df['tags'] = df['tags'].apply(ast.literal_eval)
df['encoded_tags'] = df['encoded_tags'].apply(ast.literal_eval)


labels_to_ids = {
    'O': 0,
    'GOD': 1,
    'JUS': 2,
    'CHRI': 3,
    'SACRA': 4,
    'HERESY': 5,
    'PER': 6,
    'PLACE': 7,
    'ORG': 8
    }
ids_to_labels = {v:k for k, v in labels_to_ids.items()}

labels_list = [x for x in ids_to_labels.values()]


df_prepared = df[['sentence', 'encoded_tags']]

train_df, eval_df = train_test_split(df_prepared, train_size= 0.8, random_state=42, shuffle=True)
eval_df, test_df = train_test_split(eval_df, test_size=0.5, random_state=42, shuffle=True)

tokenizer= AutoTokenizer.from_pretrained(MODEL_NER_CHECKPOINTS,  add_prefix_space=True)

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

# Transfer Learning
old_model= AutoModelForTokenClassification.from_pretrained(MODEL_NER_CHECKPOINTS) #Load the general fine-tuned model
new_model=BertTokenClassifier_With_LSTM.from_pretrained(
                                                            MODEL_ORIGINAL_CHECKPOINT,
                                                            num_labels= len(labels_list),
                                                            id2label=ids_to_labels,
                                                            label2id=labels_to_ids
                                                        )


#We just use the weights learnt by the transformer and erase the original linear classifier
new_model.bert =old_model.bert # type: ignore

# Frozing the parameters for TF
#fronzen model weights
for param in new_model.parameters(): # type: ignore
    param.requires_grad = False

# Create the new layers
new_leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Adjust the negative slope as needed


# Linear layer for feature transformation
new_dense_1 = nn.Linear(new_model.config.hidden_size, 512) # type: ignore
new_dense_2 = nn.Linear(512, 256)
# Bidirectional LSTM layer
new_bidirectional = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

# Dropout for LSTM regularization
new_dropout_lstm = nn.Dropout(0.1)  # Adjust dropout rate as needed

# Linear layer for classification
new_classifier = nn.Linear(256, new_model.config.num_labels)  # type: ignore

# Replace the existing frozen layers for the unfrozen ones

new_model.leaky_relu = new_leaky_relu # type: ignore
new_model.dense_1 = new_dense_1 # type: ignore
new_model.bidirectional = new_bidirectional # type: ignore
new_model.dropout_lstm = new_dropout_lstm # type: ignore
new_model.dense_2 = new_dense_2 # type: ignore
new_model.classifier = new_classifier # type: ignore

# Creating the torch summary
input_ids = torch.randint(0, new_model.config.vocab_size, (SUMMARY_BATCHSIZE, tokenized_inputs_train['input_ids'][0].shape[0])) # type: ignore

text_summary = str(summary(new_model, # type: ignore
            input_data=input_ids,
                       col_names=["input_size", "output_size", "num_params", "trainable"],
                       )) # type: ignore

with open(f'/home/tensorboard/Documentos/1. D4R/8. Models/summaries/sumary of TF-LSTM-{MODEL_SUMMARY_NAME}-{LEARNING_RATE}-lr-{NUM_TRAIN_EPOCHS}-epochs.txt', 'w', encoding = 'utf-8') as f:
    f.write(text_summary)
    f.close() 
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


)

trainer = Trainer(
    model=new_model, # type: ignore
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()
trainer.save_model(output_dir=OUTPUT_DIRECTORY)
