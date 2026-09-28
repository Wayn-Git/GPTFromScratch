import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        # Building the vocab: Splitting and then collecting all the unique words + sortinga and then assiging ids starting at 1

        def get_vocab(l_1: List[str], l_2: List[str]) -> List[str]:
            merged_list = l_1 + l_2

            words_list = [word for string in merged_list for word in string.split()]

            unique_words_list = sorted(set(words_list))

            return unique_words_list

        word_list = get_vocab(positive, negative)

        def build_word_map(word_list: List[str]) -> dict:
            word_map = {}
            for token_id, word in enumerate(word_list, start=1):
                word_map[word] = token_id
            return word_map 

        word_map = build_word_map(word_list)
        
        def encode_sentence(sentences):
            tensors = []
            
            for sentence in sentences: 

                tensor = []

                encode_id = [word_map[word] for word in sentence.split()]

                encoded_id_tensor = torch.tensor(encode_id, dtype=torch.float) # Converting the list into a pytorch tensor

                tensors.append(encoded_id_tensor)

            return tensors

        

        encoded_sentence = encode_sentence(positive) + encode_sentence(negative)


        output = nn.utils.rnn.pad_sequence(
            encoded_sentence,
            batch_first=True,
            padding_value=0.0
        )


        return output
