import builtins
import json

import pytest

from fastokens import Tokenizer, tiktoken_to_tokenizer_json  # noqa: E402


def _encoding():
    tiktoken = pytest.importorskip("tiktoken")
    return tiktoken.Encoding(
        name="fastokens-test",
        pat_str=r"(?s).+",
        mergeable_ranks={
            b"a": 0,
            b"b": 1,
            b"c": 2,
            b"d": 3,
            b"ab": 4,
            b"cd": 5,
            b"abcd": 6,
            b" ": 7,
            b"!": 8,
            b"c!": 9,
        },
        special_tokens={"<|end|>": 100},
    )


def test_tiktoken_to_tokenizer_json_matches_encoding() -> None:
    encoding = _encoding()
    tokenizer = Tokenizer.from_json_str(tiktoken_to_tokenizer_json(encoding))

    texts = [
        "abcd",
        "ab cd",
        "abc!",
    ]
    for text in texts:
        assert tokenizer.encode(text, add_special_tokens=False).ids == encoding.encode(text)


def test_tiktoken_to_tokenizer_json_with_encoding_name(monkeypatch: pytest.MonkeyPatch) -> None:
    tiktoken = pytest.importorskip("tiktoken")
    encoding = _encoding()
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: encoding)

    tokenizer_json = tiktoken_to_tokenizer_json("fastokens-test")
    tokenizer = Tokenizer.from_json_str(tokenizer_json)

    assert tokenizer.encode("abcd", add_special_tokens=False).ids == [6]


def test_tiktoken_to_tokenizer_json_preserves_special_tokens() -> None:
    encoding = _encoding()
    tokenizer_json = tiktoken_to_tokenizer_json(encoding)
    config = json.loads(tokenizer_json)
    special_token = "<|end|>"
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


def test_tiktoken_to_tokenizer_json_returns_none_without_optional_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_tiktoken(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("No module named 'tiktoken'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tiktoken)

    assert tiktoken_to_tokenizer_json("cl100k_base") is None
