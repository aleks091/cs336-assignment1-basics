import os
import re
import time
import json
import resource
import multiprocessing as mp
from collections import defaultdict
from typing import Dict, List, Tuple

# GPT-2 pre-tokenization regex pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file_path: str, num_chunks: int, delimiter: str = "<|endoftext|>"
) -> List[Tuple[int, int]]:
    """
    Splits a large text file into roughly equal byte ranges such that chunk boundaries
    always land on document boundaries (after a special token delimiter).
    """
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // num_chunks
    boundaries = []

    with open(file_path, "rb") as f:
        start = 0
        delimiter_bytes = delimiter.encode("utf-8")

        for i in range(num_chunks - 1):
            target = start + chunk_size
            f.seek(target)

            # Read ahead until we find the delimiter end
            buffer = f.read(4096)
            pos = buffer.find(delimiter_bytes)

            while pos == -1 and f.tell() < file_size:
                buffer = f.read(4096)
                pos = buffer.find(delimiter_bytes)

            if pos != -1:
                end = f.tell() - len(buffer) + pos + len(delimiter_bytes)
            else:
                end = file_size

            boundaries.append((start, end))
            start = end
            if start >= file_size:
                break

        if start < file_size:
            boundaries.append((start, file_size))

    return boundaries


def process_chunk(args: Tuple[str, int, int, List[str]]) -> Dict[Tuple[bytes, ...], int]:
    """
    Worker function: Reads a specific byte slice of the file, splits on special tokens,
    and runs regex pre-tokenization to return local pre-token counts.
    """
    file_path, start_byte, end_byte, special_tokens = args
    import regex as re  # Import within worker for multiprocessing safety

    with open(file_path, "rb") as f:
        f.seek(start_byte)
        chunk_bytes = f.read(end_byte - start_byte)

    chunk_str = chunk_bytes.decode("utf-8", errors="ignore")

    # Split on special tokens so they define hard boundaries and don't contribute to merge counts
    if special_tokens:
        escaped_tokens = [re.escape(st) for st in special_tokens]
        split_pattern = "|".join(escaped_tokens)
        sub_chunks = re.split(split_pattern, chunk_str)
    else:
        sub_chunks = [chunk_str]

    local_counts = defaultdict(int)
    for sub in sub_chunks:
        if not sub:
            continue
        for match in re.finditer(PAT, sub):
            pre_token = match.group()
            byte_sequence = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
            local_counts[byte_sequence] += 1

    return dict(local_counts)


def train_bpe_tinystories(
    input_path: str,
    vocab_size: int = 10000,
    special_tokens: List[str] = ["<|endoftext|>"],
    num_workers: int = None,
    vocab_out_path: str = "tinystories_vocab.json",
    merges_out_path: str = "tinystories_merges.json",
):
    start_time = time.time()
    num_workers = num_workers or max(1, mp.cpu_count() - 1)
    print(f"--- Starting BPE Training on {input_path} ---")
    print(f"Workers: {num_workers} | Target Vocab Size: {vocab_size}")

    # Step 1: Initialize Vocabulary
    vocabulary: Dict[int, bytes] = {}
    for i, token in enumerate(special_tokens):
        vocabulary[i] = token.encode("utf-8")
    offset = len(vocabulary)
    vocabulary.update({offset + i: bytes([i]) for i in range(256)})

    # Step 2: Parallel Pre-tokenization
    pretok_start = time.time()
    print("Finding document-aligned chunk boundaries...")
    boundaries = find_chunk_boundaries(
        input_path,
        num_workers,
        delimiter=special_tokens[0] if special_tokens else "<|endoftext|>",
    )

    worker_args = [(input_path, b[0], b[1], special_tokens) for b in boundaries]

    print(f"Pre-tokenizing {len(boundaries)} chunks in parallel...")
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_chunk, worker_args)

    pre_token_counts = defaultdict(int)
    for res in results:
        for seq, count in res.items():
            pre_token_counts[seq] += count

    pretok_time = time.time() - pretok_start
    print(f"Pre-tokenization completed in {pretok_time:.2f}s across {len(pre_token_counts)} unique pre-tokens.")

    # Step 3: Fast BPE Merges with Frequency Caching
    merge_start = time.time()
    iterations = max(0, vocab_size - len(vocabulary))
    merges: List[Tuple[bytes, bytes]] = []

    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)

    for word, count in pre_token_counts.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
            pair_to_words[pair].add(word)

    print(f"Computing {iterations} BPE merges...")
    for step in range(iterations):
        if not pair_counts:
            break

        # Max count with lexicographical tie-breaker
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        byte1, byte2 = best_pair
        merged_token = byte1 + byte2

        vocabulary[len(vocabulary)] = merged_token
        merges.append(best_pair)

        words_to_process = list(pair_to_words[best_pair])
        del pair_counts[best_pair]
        del pair_to_words[best_pair]

        for word in words_to_process:
            count = pre_token_counts[word]
            del pre_token_counts[word]

            # Subtract old pairs
            for pair in zip(word, word[1:]):
                if pair in pair_counts:
                    pair_counts[pair] -= count
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                if pair in pair_to_words and word in pair_to_words[pair]:
                    pair_to_words[pair].remove(word)
                    if not pair_to_words[pair]:
                        del pair_to_words[pair]

            # Replace best_pair in word
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == byte1 and word[i + 1] == byte2:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)

            # Add new pairs
            pre_token_counts[new_word] += count
            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] += count
                pair_to_words[pair].add(new_word)

    merge_time = time.time() - merge_start
    total_time = time.time() - start_time

    # Memory Usage (Peak RSS)
    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # MB on Linux

    # Step 4: Serialize Vocabulary & Merges
    print(f"Saving vocabulary to {vocab_out_path} and merges to {merges_out_path}...")

    serialized_vocab = {idx: token.decode("latin-1") for idx, token in vocabulary.items()}
    serialized_merges = [[p[0].decode("latin-1"), p[1].decode("latin-1")] for p in merges]

    with open(vocab_out_path, "w", encoding="utf-8") as f:
        json.dump(serialized_vocab, f, indent=2, ensure_ascii=False)

    with open(merges_out_path, "w", encoding="utf-8") as f:
        json.dump(serialized_merges, f, indent=2, ensure_ascii=False)

    # Step 5: Answer Problem (train_bpe_tinystories) Deliverables
    longest_token_bytes = max(vocabulary.values(), key=len)
    longest_token_str = longest_token_bytes.decode("utf-8", errors="replace")

    print("\n" + "=" * 50)
    print("TRAINING SUMMARY & DELIVERABLES")
    print("=" * 50)
    print(f"Total Wall-Clock Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"  - Pre-tokenization time: {pretok_time:.2f}s")
    print(f"  - Merging time:          {merge_time:.2f}s")
    print(f"Peak Memory Usage:      {max_rss_mb:.2f} MB ({max_rss_mb / 1024:.2f} GB)")
    print(f"Longest Token Length:   {len(longest_token_bytes)} bytes")
    print(f"Longest Token Value:    '{longest_token_str}'")
    print("=" * 50)


if __name__ == "__main__":
    DATASET_PATH = "../data/TinyStoriesV2-GPT4-train.txt"

    if os.path.exists(DATASET_PATH):
        train_bpe_tinystories(
            input_path=DATASET_PATH,
            vocab_size=10000,
            special_tokens=["<|endoftext|>"],
        )
    else:
        print(f"File '{DATASET_PATH}' not found! Please check your dataset path.")
