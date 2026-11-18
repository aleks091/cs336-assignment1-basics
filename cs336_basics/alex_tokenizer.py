# Different kinds of tokenizers: 
# Byte-level tokenizers, word-level tokenizers and sub-word tokenizers. 

# The process of constructing the BPE tokenizer vocabulary 
# is known as training the BPE tokenizer.

from collections import defaultdict
from typing import Dict, List, Tuple
import regex as re
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Byte-level Byte-Pair Encoding Tokenizer
def alex_tokenizer(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,):
    
    # Step 1: Vocabulary initialization
    vocabulary: Dict[int, bytes] = {}

    for i, token in enumerate(special_tokens):
        vocabulary[i] = token.encode("utf-8")

    offset_index = len(vocabulary)
    vocabulary.update({offset_index + i: bytes([i]) for i in range(256)})

    # Step 2: Pre-tokenization
    print(f"Reading from CORPUS path: {input_path}")
    content = ""
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunks = []
    if special_tokens:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        split_pattern = "|".join(escaped_tokens)
        # Splits on special tokens so they act as hard boundaries and don't contribute to merge counts
        chunks = re.split(split_pattern, content)
    else: 
        chunks = [content]

    pre_token_counts = defaultdict(int)
    for chunk in chunks:
        if not chunk:
            continue
            
        for match in re.finditer(PAT, chunk):
            pre_token = match.group()
            byte_sequence = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
            pre_token_counts[byte_sequence] += 1
    
    merges: List[Tuple[bytes, bytes]] = []

    # Step 3: Compute BPE Merges (Optimized with Caching)
    iterations = max(0, vocab_size - len(vocabulary))
    
    # Pre-compute pair frequencies and the reverse mapping from pairs to the words containing them
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    
    for word, count in pre_token_counts.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
            pair_to_words[pair].add(word)
            
    for _ in range(iterations):
        if not pair_counts:
            break
            
        # Find the most frequent pair. 
        # Tie-breaker: sorts by frequency first, then lexicographically.
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        
        # Unpack the tuple into its two individual bytes
        byte1, byte2 = best_pair
        
        # Concatenate the two bytes objects into a single merged token
        merged_token = byte1 + byte2
        
        # 1) Add the new merged token to our vocabulary
        vocabulary[len(vocabulary)] = merged_token
        
        # 2) Add the 2 merged tokens into merges
        merges.append(best_pair)
        
        # 3) Update pre_token_counts and pair_counts incrementally
        words_to_process = list(pair_to_words[best_pair])
        
        # Clean up the merged pair from our tracking dictionaries
        del pair_counts[best_pair]
        del pair_to_words[best_pair]
        
        for word in words_to_process:
            count = pre_token_counts[word]
            del pre_token_counts[word]
            
            # Subtract counts of all pairs in the old un-merged word
            for pair in zip(word, word[1:]):
                if pair in pair_counts:
                    pair_counts[pair] -= count
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                
                if pair in pair_to_words and word in pair_to_words[pair]:
                    pair_to_words[pair].remove(word)
                    if not pair_to_words[pair]:
                        del pair_to_words[pair]
            
            # Merge the best_pair in the word
            new_word = []
            i = 0
            while i < len(word):
                # Use our unpacked bytes for the comparison!
                if i < len(word) - 1 and word[i] == byte1 and word[i+1] == byte2:
                    new_word.append(merged_token)
                    i += 2 # Skip the next token since we just merged it
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            
            # Add the new_word to tracking and update pairs
            pre_token_counts[new_word] += count
            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] += count
                pair_to_words[pair].add(new_word)

    return (vocabulary, merges)