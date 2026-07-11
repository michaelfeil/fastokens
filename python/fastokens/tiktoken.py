from __future__ import annotations

import json
from typing import Any


def _byte_to_unicode() -> dict[int, str]:
    table: dict[int, str] = {}
    next_codepoint = 256
    for byte in range(256):
        should_use_direct_mapping = 33 <= byte <= 126 or 0xA1 <= byte <= 0xAC or byte >= 0xAE
        if should_use_direct_mapping:
            table[byte] = chr(byte)
        else:
            table[byte] = chr(next_codepoint)
            next_codepoint += 1
    return table


_BYTE_ENCODER = _byte_to_unicode()


def _token_bytes_to_string(token: bytes) -> str:
    return "".join(_BYTE_ENCODER[byte] for byte in token)


def _extract_encoding(encoding: Any) -> Any:
    if isinstance(encoding, str):
        try:
            import tiktoken
        except ImportError as exc:
            raise ValueError(
                "`tiktoken` is required when passing an encoding name. "
                "Install it with `pip install tiktoken`."
            ) from exc
        return tiktoken.get_encoding(encoding)
    return encoding


def _extract_vocab_and_merges(mergeable_ranks: dict[bytes, int]) -> tuple[dict[str, int], list[list[str]]]:
    vocab = {
        _token_bytes_to_string(token): rank
        for token, rank in sorted(mergeable_ranks.items(), key=lambda item: item[1])
    }

    merge_candidates: list[tuple[int, int, int, bytes, bytes]] = []
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        local: list[tuple[int, int, bytes, bytes]] = []
        for index in range(1, len(token)):
            left = token[:index]
            right = token[index:]
            if left in mergeable_ranks and right in mergeable_ranks:
                local.append((mergeable_ranks[left], mergeable_ranks[right], left, right))
        local.sort(key=lambda item: (item[0], item[1]))
        merge_candidates.extend((rank, left_rank, right_rank, left, right) for left_rank, right_rank, left, right in local)

    merge_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    merges = [[_token_bytes_to_string(left), _token_bytes_to_string(right)] for _, _, _, left, right in merge_candidates]
    return vocab, merges


def tiktoken_to_tokenizer_json(encoding: Any, *, pretty: bool = False) -> str:
    """
    Convert a ``tiktoken`` encoding to a Hugging Face ``tokenizer.json`` string.

    ``encoding`` may be either a ``tiktoken.Encoding`` instance or an encoding
    name accepted by ``tiktoken.get_encoding``. The returned JSON can be passed
    directly to ``fastokens.Tokenizer.from_json_str``.
    """
    encoding = _extract_encoding(encoding)
    try:
        mergeable_ranks = encoding._mergeable_ranks
        pattern = encoding._pat_str
        special_tokens = encoding._special_tokens
    except AttributeError as exc:
        raise TypeError(
            "expected a tiktoken.Encoding or encoding name with "
            "_mergeable_ranks, _pat_str, and _special_tokens"
        ) from exc

    vocab, merges = _extract_vocab_and_merges(mergeable_ranks)
    added_tokens = [
        {
            "id": token_id,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token, token_id in sorted(special_tokens.items(), key=lambda item: item[1])
    ]

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": pattern},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": merges,
        },
    }
    if pretty:
        return json.dumps(tokenizer_json, indent=2, ensure_ascii=False)
    return json.dumps(tokenizer_json, ensure_ascii=False)
