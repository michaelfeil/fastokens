import json

import pytest

tiktoken = pytest.importorskip("tiktoken")

from fastokens import Tokenizer, tiktoken_to_tokenizer_json  # noqa: E402


def test_tiktoken_to_tokenizer_json_matches_cl100k_base() -> None:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenizer = Tokenizer.from_json_str(tiktoken_to_tokenizer_json(encoding))

    texts = [
        "Hello, world!",
        "it's 2026, and tokenization is fast",
        "unicode: café 火 🚀",
    ]
    for text in texts:
        assert tokenizer.encode(text, add_special_tokens=False).ids == encoding.encode(text)


def test_tiktoken_to_tokenizer_json_accepts_encoding_name_and_special_tokens() -> None:
    tokenizer_json = tiktoken_to_tokenizer_json("cl100k_base")
    config = json.loads(tokenizer_json)
    encoding = tiktoken.get_encoding("cl100k_base")
    special_token = "<|endoftext|>"
    special_id = encoding._special_tokens[special_token]

    assert {
        "id": special_id,
        "content": special_token,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    } in config["added_tokens"]

    tokenizer = Tokenizer.from_json_str(tokenizer_json)
    assert tokenizer.encode(special_token, add_special_tokens=False).ids == [special_id]
