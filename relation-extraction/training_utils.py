import torch
import torch.utils.data 
from typing import Optional, List
from transformers import AutoTokenizer, BatchEncoding

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
    



import torch
from typing import List, Optional
from transformers import AutoTokenizer, BatchEncoding

def tokenize_and_position_sequence(
    sequences: Optional[List[List[str]]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    labels: Optional[List[List[int]]] = None,
    special_tokens: Optional[List[int]] = [6, 7, 8, 9, 10],
    initial_shifting_value: Optional[int] = 50,
    incremental: Optional[int] = 50
) -> BatchEncoding:
    """
    Tokenize and position a sequence of text.

    Args:
        sequences: A list of lists of strings, where each inner list is a sequence of text.
        tokenizer: An AutoTokenizer instance.
        labels: A list of lists of integers, where each inner list is a label for the corresponding sequence.
        special_tokens: A list of special token IDs. Defaults to [6, 7, 8, 9, 10].
        initial_shifting_value: Initial value for shifting. Defaults to 10.

    Returns:
        A BatchEncoding instance with the tokenized input IDs, attention masks, and sequence positions.
    """
    shifting_value = initial_shifting_value
    all_tensors_positive = False
    
    while not all_tensors_positive:
        tokenized_inputs = tokenizer(
            sequences,
            truncation=True,
            is_split_into_words=True,
            padding=True,
            add_special_tokens=False,
            return_tensors='pt'
        ) # type: ignore

        max_length = tokenized_inputs['input_ids'].shape[1]
        sequence_positions = []

        for tensor in tokenized_inputs['input_ids']:
            indices = [i for i, token in enumerate(tensor) if token in special_tokens]
            if len(indices) != 2:
                raise ValueError(f"Expected exactly 2 special tokens, but found {len(indices)}.")

            entity_pos, _ = indices  # Assuming the first special token is the entity position
            shift_value = shifting_value - entity_pos # type: ignore

            pos1 = torch.arange(-entity_pos, max_length - entity_pos) + shift_value
            pos2 = torch.arange(-indices[1], max_length - indices[1]) + shift_value

            sequence_positions.append(torch.stack([pos1, pos2]))

        sequence_positions = torch.stack(sequence_positions)
        tokenized_inputs['sequence_positions'] = sequence_positions
        tokenized_inputs['labels'] = torch.tensor(labels)

        # Check if all tensors have positive values
        all_tensors_positive = True
        
        for i, tensor in enumerate(tokenized_inputs['sequence_positions'][:, 1]):
            if min(tensor) <= 0:
                all_tensors_positive = False
                break
        
        if not all_tensors_positive:
            shifting_value += incremental # type: ignore
            print(f'Not all are positive, shifting the value to {shifting_value}')
    print(f'Final shifting_value: {shifting_value}')

    return tokenized_inputs
