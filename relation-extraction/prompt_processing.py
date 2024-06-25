import numpy as np
from numba import njit
import regex as re
import json
from typing import Optional, List, Dict
import pandas as pd

import json
from typing import Optional, List, Dict
import pandas as pd
import re

def process_re_prompt(
    df_reduced: Optional[pd.DataFrame] = None,
    special_tokens: Optional[List] = None,
) -> List[Dict[str, List[str]]]:
    """
    Process a reduced DataFrame to create prompted sentences for sequence-to-sequence models.

    Parameters:
    df_reduced (Optional[pd.DataFrame]): The reduced DataFrame containing the data to be processed. Defaults to None.
    special_tokens (Optional[List]): A list of special tokens to be used in the prompted sentences. Defaults to None, which sets it to ['[ORG]', '[PERSON]', '[PERSON_REFERENCE]', '[PLACE]'].

    Returns:
    List[Dict[str, List[str]]]: A list of dictionaries, where each dictionary contains a prompted sentence and its corresponding label.

    Notes:
    This function assumes that the input DataFrame has the following columns: 'Sentence', 'Entity_1', 'Entity_1_type', 'Entity_2', 'Entity_2_type', and 'Label'.
    The function cleans the sentences, replaces entities with their corresponding types, creates prompted sentences, tokenizes them, and returns a list of dictionaries with the prompted sentences and labels.
    """
    new_df_list = []
    if special_tokens is None:
        special_tokens = ['[ORG]', '[PERSON]', '[PERSON_REFERENCE]', '[PLACE]']

    for index, item in df_reduced.iterrows():# type: ignore
        clean_sentence = re.sub(r'\[.*?\]', '', item['Sentence'])
        enti_1 = item['Entity_1']
        enti_mask_1 = item['Entity_1_type']
        
        if item['Entity_2'] != 'CONTEXT':
            enti_2 = item['Entity_2']
            enti_mask_2 = item['Entity_2_type']
            prompted_sentence = f"[CLS] {clean_sentence.replace(enti_1, f' [{enti_mask_1}] ',1).replace(enti_2, f' [{enti_mask_2}] ', 1)} [SEP] {item['Entity_1']} [SEP] {item['Entity_2']} [SEP]"
        else:
            prompted_sentence = f"[CLS] {clean_sentence.replace(enti_1, f' [{enti_mask_1}] ', 1)} [CONTEXT] [SEP] {item['Entity_1']} [SEP] CONTEXT [SEP]"
        
        prompted_sentence = re.sub(r'\s+', ' ', prompted_sentence)
        tokenized_for_sequence = prompted_sentence.split()
        label = item['Label']

        new_df_list.append({
            'Prompted_sentence': tokenized_for_sequence,
            'Label': label
        })

    return new_df_list

def process_re_prompt_blackbox(df_reduced: Optional[pd.DataFrame] = None, special_tokens: Optional[List] = None) -> List[Dict[str, List[str]]]:
    """
    Process a reduced DataFrame to create prompted sentences for sequence-to-sequence models. This function was improved with BlackBox Code AI.

    Parameters:
    df_reduced (Optional[pd.DataFrame]): The reduced DataFrame containing the data to be processed. Defaults to None.
    special_tokens (Optional[List]): A list of special tokens to be used in the prompted sentences. Defaults to None, which sets it to ['[ORG]', '[PERSON]', '[PERSON_REFERENCE]', '[PLACE]'].

    Returns:
    List[Dict[str, List[str]]]: A list of dictionaries, where each dictionary contains a prompted sentence and its corresponding label.

    Notes:
    This function assumes that the input DataFrame has the following columns: 'Sentence', 'Entity_1', 'Entity_1_type', 'Entity_2', 'Entity_2_type', and 'Label'.
    The function cleans the sentences, replaces entities with their corresponding types, creates prompted sentences, tokenizes them, and returns a list of dictionaries with the prompted sentences and labels.
    """
    if special_tokens is None:
        special_tokens = ['[ORG]', '[PERSON]', '[PERSON_REFERENCE]', '[PLACE]']

    # Vectorized operations
    df_reduced['clean_sentence'] = df_reduced['Sentence'].str.replace(r'\[.*?\]', '') # type: ignore
    df_reduced['entity_1_masked'] = df_reduced.apply(lambda row: row['clean_sentence'].replace(row['Entity_1'], f' [{row["Entity_1_type"]}] ', 1), axis=1) # type: ignore
    df_reduced['entity_2_masked'] = df_reduced.apply(lambda row: row['entity_1_masked'].replace(row['Entity_2'], f' [{row["Entity_2_type"]}] ', 1) if row['Entity_2'] != 'CONTEXT' else row['entity_1_masked'], axis=1) # type: ignore

    # Create prompted sentence using f-strings
    df_reduced['prompted_sentence'] = df_reduced.apply(lambda row: f"[CLS] {row['entity_2_masked']} [SEP] {row['Entity_1']} [SEP] {row['Entity_2'] if row['Entity_2'] != 'CONTEXT' else 'CONTEXT'} [SEP]", axis=1)# type: ignore

    # Tokenize prompted sentence
    df_reduced['tokenized_sentence'] = df_reduced['prompted_sentence'].str.split() # type: ignore

    # Create list of dictionaries
    new_df_list = df_reduced.apply(lambda row: {'Prompted_sentence': row['tokenized_sentence'], 'Label': row['Label']}, axis=1).tolist() # type: ignore

    return new_df_list