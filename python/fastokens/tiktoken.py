from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

DEFAULT_TIKTOKEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
    r"\s+(?!\S)|\s+"
)
PRINTABLE_ASCII_START = 33
PRINTABLE_ASCII_END = 126
LATIN1_DIRECT_START = 0xA1
LATIN1_DIRECT_GAP_END = 0xAC
LATIN1_DIRECT_RESUME = 0xAE


class _MergeCandidate(NamedTuple):
    rank: int
    left_rank: int
    right_rank: int
    left: bytes
    right: bytes


def _byte_to_unicode() -> dict[int, str]:
    table: dict[int, str] = {}
    next_codepoint = 256
    for byte in range(256):
        # Match the GPT-2 byte-level alphabet: printable ASCII plus most
        # Latin-1 bytes are kept as-is, while whitespace/control bytes map
        # to code points above the byte range.
        should_use_direct_mapping = (
            PRINTABLE_ASCII_START <= byte <= PRINTABLE_ASCII_END
            or LATIN1_DIRECT_START <= byte <= LATIN1_DIRECT_GAP_END
            or byte >= LATIN1_DIRECT_RESUME
        )
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
        except ImportError:
            return None
        return tiktoken.get_encoding(encoding)
    return encoding


def _extract_vocab_and_merges(mergeable_ranks: dict[bytes, int]) -> tuple[dict[str, int], list[list[str]]]:
    vocab = {
        _token_bytes_to_string(token): rank
        for token, rank in sorted(mergeable_ranks.items(), key=lambda item: item[1])
    }

    merge_candidates: list[_MergeCandidate] = []
    def _get_rank_pair_for_sorting(split: tuple[int, int, bytes, bytes]) -> tuple[int, int]:
        left_rank, right_rank, _, _ = split
        return left_rank, right_rank

    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        split_candidates: list[tuple[int, int, bytes, bytes]] = []
        for index in range(1, len(token)):
            left = token[:index]
            right = token[index:]
            if left in mergeable_ranks and right in mergeable_ranks:
                split_candidates.append((mergeable_ranks[left], mergeable_ranks[right], left, right))
        split_candidates.sort(key=_get_rank_pair_for_sorting)
        merge_candidates.extend(
            _MergeCandidate(
                rank=rank,
                left_rank=left_rank,
                right_rank=right_rank,
                left=left,
                right=right,
            )
            for left_rank, right_rank, left, right in split_candidates
        )

    merge_candidates.sort(key=lambda item: (item.rank, item.left_rank, item.right_rank))
    merges = [
        [_token_bytes_to_string(candidate.left), _token_bytes_to_string(candidate.right)]
        for candidate in merge_candidates
    ]
    return vocab, merges


def _normalise_special_tokens(special_tokens: Mapping[str, int] | None) -> dict[str, int]:
    if special_tokens is None:
        return {}
    return dict(special_tokens)


def _config_special_tokens(config: dict[str, Any]) -> dict[str, int]:
    special_tokens: dict[str, int] = {}
    added_tokens_decoder = config.get("added_tokens_decoder", {})
    if isinstance(added_tokens_decoder, dict):
        for token_id, token_config in added_tokens_decoder.items():
            try:
                token_id_int = int(token_id)
            except (TypeError, ValueError):
                continue
            if isinstance(token_config, dict):
                content = token_config.get("content")
                is_special = token_config.get("special", False)
            else:
                content = token_config
                is_special = False
            if isinstance(content, str) and is_special:
                special_tokens[content] = token_id_int
    return special_tokens


def _load_model_config(model: str | Path) -> tuple[str, dict[str, Any]]:
    if isinstance(model, str) and urlparse(model).scheme in {"http", "https", "s3", "gs", "az", "file"}:
        return model, {}
    model_path = Path(model)
    if model_path.is_dir():
        config_path = model_path / "tokenizer_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        return str(model_path / "tiktoken.model"), config
    return str(model), {}


def _tokenizer_json_from_parts(
    mergeable_ranks: dict[bytes, int],
    pattern: str,
    special_tokens: Mapping[str, int] | None,
    *,
    pretty: bool = False,
) -> str:
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
        for token, token_id in sorted(_normalise_special_tokens(special_tokens).items(), key=lambda item: item[1])
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


def tiktoken_model_to_tokenizer_json(
    model: str | Path,
    *,
    pattern: str | None = None,
    special_tokens: Mapping[str, int] | None = None,
    pretty: bool = False,
) -> str | None:
    """
    Convert a ``tiktoken.model`` BPE file to a Hugging Face ``tokenizer.json`` string.

    ``model`` may point to a local ``tiktoken.model`` file, a URL accepted by
    ``tiktoken.load.load_tiktoken_bpe``, or a local directory containing
    ``tiktoken.model`` and optionally ``tokenizer_config.json``. Passing a model
    path returns ``None`` if the optional ``tiktoken`` package is not installed.
    """
    try:
        from tiktoken.load import load_tiktoken_bpe
    except ImportError:
        return None

    model_path, config = _load_model_config(model)
    if pattern is None:
        config_pattern = config.get("pat_str")
        if not isinstance(config_pattern, str):
            config_pattern = config.get("pattern")
        pattern = config_pattern if isinstance(config_pattern, str) else DEFAULT_TIKTOKEN_PATTERN
    special_tokens = _config_special_tokens(config) if special_tokens is None else special_tokens
    try:
        mergeable_ranks = load_tiktoken_bpe(model_path)
    except ValueError as exc:
        raise ValueError(f"invalid tiktoken model {model_path!r}: {exc}") from exc
    return _tokenizer_json_from_parts(mergeable_ranks, pattern, special_tokens, pretty=pretty)


def tiktoken_to_tokenizer_json(encoding: Any, *, pretty: bool = False) -> str | None:
    """
    Convert a ``tiktoken`` encoding to a Hugging Face ``tokenizer.json`` string.

    ``encoding`` may be either a ``tiktoken.Encoding`` instance or an encoding
    name accepted by ``tiktoken.get_encoding``. Passing an encoding name returns
    ``None`` if the optional ``tiktoken`` package is not installed. The returned
    JSON can be passed directly to ``fastokens.Tokenizer.from_json_str``.
    """
    encoding = _extract_encoding(encoding)
    if encoding is None:
        return None
    try:
        mergeable_ranks = encoding._mergeable_ranks
        pattern = encoding._pat_str
        special_tokens = encoding._special_tokens
    except AttributeError as exc:
        raise TypeError(
            "expected a tiktoken.Encoding or encoding name with "
            "_mergeable_ranks, _pat_str, and _special_tokens"
        ) from exc

    return _tokenizer_json_from_parts(mergeable_ranks, pattern, special_tokens, pretty=pretty)
