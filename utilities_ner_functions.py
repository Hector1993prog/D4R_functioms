import torch
import evaluate
import numpy as np

import torch.utils.data

class baseDataset(torch.utils.data.Dataset): 
    """
    Base dataset class for handling encodings.

    Attributes:
        encodings (dict): A dictionary containing the encodings for the dataset.

    Methods:
        __getitem__(self, idx: int) -> dict:
            Returns a dictionary containing the cloned and detached values at the specified index.

        __len__(self) -> int:
            Returns the length of the input_ids in the encodings.
    """
    def __init__(self, encodings):
        
        self.encodings = encodings

    def __getitem__(self, idx):
       
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        
        return len(self.encodings.input_ids)
    

##########################################################################################################
def tokenize_and_align_labels_black_box(dataframe,
                                        tokenizer,
                                        token_column= 'Tokens',
                                        tag_column = 'Tags'
                                        ):
    """
    Tokenize inputs using the provided tokenizer and align labels accordingly. Improved by Black Box AI coder.

    Args:
        dataframe (pandas.DataFrame): DataFrame containing 'tokens' and 'ner_tags' columns.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use for tokenization.

    Returns:
        dict: Tokenized inputs with aligned labels as PyTorch tensors.
    """
    # Tokenize inputs using the provided tokenizer
    tokenized_inputs = tokenizer(
        dataframe[token_column].tolist(),
        truncation=True,
        is_split_into_words=True,
        padding=True,
        max_length=tokenizer.model_max_length,
        return_tensors='pt'
    )
    labels = []
    for i, label in enumerate(dataframe[tag_column].tolist()):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        label_ids = [-100 if word_id is None else label[word_id] if word_id < len(label) else -100 for word_id in word_ids]
        labels.append(label_ids)

    # Convert lists to PyTorch tensors
    tokenized_inputs["labels"] = torch.tensor(labels)

    return tokenized_inputs

#####################################################################################################
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