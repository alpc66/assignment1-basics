"""A BPE tokenizer implementation"""

import os
import time
import pstats
import heapq
import cProfile
import multiprocessing
import regex as re
from pstats import SortKey
from typing import BinaryIO
from icecream import ic
from collections import Counter, defaultdict

# Regex for coarse tokenization
PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
base = os.path.dirname(os.path.abspath(__file__))


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train bytes-level BPE from a text file.

    Args:
        input_path (str): Path to the input text file.
        vocab_size (int): Desired vocabulary size.
        special_tokens (list[str]): List of special tokens to include in the vocabulary.

    Returns:
        [vocab, merges]: A tuple containing the vocabulary and merge operations.
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: The vocabulary and merge operations.
    """
    train_start_time = time.time()
    tokens = [bytes(b) for b in range(256)] + [
        st.encode("utf-8") for st in special_tokens
    ]
    vocab_dict = {idx: token for idx, token in enumerate(tokens)}

    # 1. pretokenize
    print("Pretokenizing...")
    start_time = time.time()
    freqs = pretokenize(input_path, special_tokens)
    print(f"Pretokenization took {time.time() - start_time:.2f} seconds")
    # 2. get pair
    print("Getting pairs...")
    start_time = time.time()
    pairs, pair2key = get_pairs(freqs)
    print(f"Getting pairs took {time.time() - start_time:.2f} seconds")
    # 3. merge tokens


def call_pretokenize_chunk(args):
    """Call pretokenize_chunk with the given arguments to unpack tuple."""
    return pretokenize_chunk(*args)


def pretokenize(input_path: str, special_tokens: list[str]) -> list[bytes]:
    """Pretokenize text into a list of byte tokens.

    Args:
        text (str): Input text string.

    Returns:
        list[bytes]: List of byte tokens.
    """

    special_pattern = (
        re.compile("|".join(re.escape(st) for st in special_tokens))
        if special_tokens
        else None
    )
    with open(input_path, "rb") as f:
        total = Counter()
        num_processes = multiprocessing.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        ic(boundaries)
        stream = ((chunk, special_pattern) for chunk in iter_chunk_bytes(f, boundaries))
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.imap_unordered(call_pretokenize_chunk, stream, chunksize=4)
            for freqs in results:
                total.update(freqs)
        # ic(total.most_common(30))
        return total


def iter_chunk_bytes(f: BinaryIO, boundaries: list[int]):
    """Yield chunks of bytes from a file given the boundaries."""
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        yield chunk


def pretokenize_chunk(
    chunk: str, special_pattern: re.Pattern | None
) -> dict[tuple[bytes], int]:
    """Pretokenize a chunk of text into a list of byte tokens."""

    freqs = Counter()
    chunks = re.split(special_pattern, chunk) if special_pattern else [chunk]

    for sub_chunk in chunks:
        for match in re.finditer(PAT, sub_chunk):
            match_bytes = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
            freqs[match_bytes] += 1
    return freqs


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def get_pairs(
    freqs: dict[tuple[bytes], int],
) -> tuple[Counter, dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    """Get counts of all symbol pairs in the dataset."""
    pairs = Counter()
    pair2key = defaultdict(set)
    for symbols, freq in freqs.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] += freq
            pair2key[pair].add(symbols)
    return pairs, pair2key


def bpe_encode():
    pass


def analyze_profile(name="bpe.prof"):
    NUM = 10
    # load stats file
    p = pstats.Stats(name)

    # filter
    p.strip_dirs()

    # sort
    print("\n=== Top functions by cumulative time ===")
    p.sort_stats(SortKey.CUMULATIVE).print_stats(NUM)

    print("\n=== Top functions by total time ===")
    p.sort_stats(SortKey.TIME).print_stats(NUM)

    print("\n=== Top functions by number of calls ===")
    p.sort_stats(SortKey.CALLS).print_stats(NUM)


if __name__ == "__main__":
    file_path = os.path.join(base, "..", "data", "TinyStoriesV2-GPT4-valid.txt")
    assert os.path.exists(file_path), f"File {file_path} does not exist."
    cProfile.run(
        "train_bpe(os.path.join(base, '..', 'data', 'TinyStoriesV2-GPT4-valid.txt'), 10000, ['<|endoftext|>'])",
        "bpe.prof",
    )
    # analyze_profile()
    # pretokenize_chunk("hello<|endoftext|>world", re.compile(re.escape("<|endoftext|>")))
