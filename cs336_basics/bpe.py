"""A BPE tokenizer implementation"""

import re
import os
import cProfile
import multiprocessing
from typing import BinaryIO
from icecream import ic

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
    tokens = [bytes(b) for b in range(256)] + [
        st.encode("utf-8") for st in special_tokens
    ]
    vocab_dict = {idx: token for idx, token in enumerate(tokens)}

    # 1. pretokenize
    pretokenize(input_path, special_tokens)
    # 2. remove special tokens from the text
    # 3. merge tokens


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
        num_processes = multiprocessing.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        ic(boundaries)


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


def bpe_encode():
    pass


if __name__ == "__main__":
    train_bpe(
        os.path.join(base, "..", "data", "TinyStoriesV2-GPT4-valid.txt"),
        10000,
        ["<|endoftext|>"],
    )
