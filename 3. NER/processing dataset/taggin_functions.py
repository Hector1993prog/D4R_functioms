from typing import List, Dict

def tagging_ner(
        to_tag: List[str],
        dictionary_symbols: Dict[str, str]
)-> List[str]:

    """
    This function performs Named Entity Recognition (NER) tagging based
    on a given list of tokens and a dictionary of symbols and their corresponding tags.

    Parameters:
        to_tag (List[str]): A list of tokens to be tagged for NER.
        dictionary_symbols (Dict[str, int]): A dictionary where keys are symbols to be matched
        in the tokens and values are the corresponding tags.

    Returns:
        List[int]: A list of tags assigned to each token based on the dictionary_symbols.
        If a token does not match any symbol, it is assigned a tag of 0.

    """

    output_list = ['O'] * len(to_tag)
    inside_marker = False

    for i, token in enumerate(to_tag):
        for k, v in dictionary_symbols.items():
            if token.startswith(k) and token.endswith(k):
                inside_marker = False
                output_list[i] = v 
                break
            elif token.startswith(k):
                inside_marker = True
                inside_value = v
                output_list[i] = v 
                break
            
            elif token.endswith(k):
                inside_marker = False
                output_list[i] = v 
                break
            elif inside_marker:
                
                output_list[i] = inside_value
    
    return output_list 


def encoding_ner_tags(tags_list: List[str], encoding_dict: Dict[str, int], no_entity_tag: int = 0) -> List[int]:
    """
    Encodes a list of tags using a given encoding dictionary.

    This function takes a list of tags (strings) and an encoding dictionary as input,
    and returns a new list where each tag is replaced with its corresponding integer
    value from the encoding dictionary.

    Args:
        tags_list (List[str]): A list of tags to be encoded.
        encoding_dict (Dict[str, int]): A dictionary that maps each tag to its
            corresponding integer value.

    Returns:
        List[int]: A new list where each tag is replaced with its corresponding
            integer value from the encoding dictionary.
    """
    encoded_list = [no_entity_tag] * len(tags_list)
    for i, tag in enumerate(tags_list):
        for k, v in encoding_dict.items():
            if k == tag:
                encoded_list[i] = v
    return encoded_list

