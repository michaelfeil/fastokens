import base64
import builtins
import json

import pytest

from fastokens import Tokenizer, tiktoken_model_to_tokenizer_json, tiktoken_to_tokenizer_json


@pytest.fixture
def hide_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_without_tiktoken(name, *args, **kwargs):
        if name == "tiktoken" or name.startswith("tiktoken."):
            raise ImportError("No module named 'tiktoken'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tiktoken)


def _create_test_encoding():
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


def _write_tiktoken_model(path, encoding) -> None:
    path.write_text(
        "\n".join(
            f"{base64.b64encode(token).decode()} {rank}"
            for token, rank in encoding._mergeable_ranks.items()
        )
    )


def test_tiktoken_to_tokenizer_json_matches_encoding() -> None:
    encoding = _create_test_encoding()
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
    encoding = _create_test_encoding()
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: encoding)

    tokenizer_json = tiktoken_to_tokenizer_json("fastokens-test")
    tokenizer = Tokenizer.from_json_str(tokenizer_json)

    assert tokenizer.encode("abcd", add_special_tokens=False).ids == [6]


def test_tiktoken_to_tokenizer_json_preserves_special_tokens() -> None:
    encoding = _create_test_encoding()
    tokenizer_json = tiktoken_to_tokenizer_json(encoding)
    config = json.loads(tokenizer_json)
    special_token = "<|end|>"
    special_id = 100

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
    hide_tiktoken,
) -> None:
    assert tiktoken_to_tokenizer_json("cl100k_base") is None


def test_tiktoken_model_to_tokenizer_json_matches_model_file(tmp_path) -> None:
    tiktoken =     pytest.importorskip("tiktoken")
    encoding = _create_test_encoding()
    model_path = tmp_path / "tiktoken.model"
    _write_tiktoken_model(model_path, encoding)

    tokenizer_json = tiktoken_model_to_tokenizer_json(
        model_path,
        pattern=encoding._pat_str,
        special_tokens={"<|end|>": 100},
    )
    tokenizer = Tokenizer.from_json_str(tokenizer_json)

    for text in ["abcd", "ab cd", "abc!"]:
        assert tokenizer.encode(text, add_special_tokens=False).ids == encoding.encode(text)
    assert tokenizer.encode("<|end|>", add_special_tokens=False).ids == [100]


def test_tiktoken_model_to_tokenizer_json_reads_model_directory(tmp_path) -> None:
    tiktoken = pytest.importorskip("tiktoken")
    encoding = _create_test_encoding()
    _write_tiktoken_model(tmp_path / "tiktoken.model", encoding)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pat_str": encoding._pat_str,
                "added_tokens_decoder": {
                    "100": {
                        "content": "<|end|>",
                        "special": True,
                    }
                },
            }
        )
    )

    tokenizer_json = tiktoken_model_to_tokenizer_json(tmp_path)
    tokenizer = Tokenizer.from_json_str(tokenizer_json)

    assert tokenizer.encode("abcd", add_special_tokens=False).ids == [6]
    assert tokenizer.encode("<|end|>", add_special_tokens=False).ids == [100]


def test_tiktoken_model_to_tokenizer_json_returns_none_without_optional_tiktoken(
    hide_tiktoken,
) -> None:
    assert tiktoken_model_to_tokenizer_json("tiktoken.model") is None


def test_tiktoken_model_to_tokenizer_json_raises_for_missing_file(tmp_path) -> None:
    pytest.importorskip("tiktoken")

    with pytest.raises(FileNotFoundError):
        tiktoken_model_to_tokenizer_json(tmp_path / "missing.model")


def test_tiktoken_model_to_tokenizer_json_raises_for_invalid_file(tmp_path) -> None:
    pytest.importorskip("tiktoken")
    model_path = tmp_path / "tiktoken.model"
    model_path.write_text("not-a-valid-model-line")

    with pytest.raises(ValueError, match="invalid tiktoken model"):
        tiktoken_model_to_tokenizer_json(model_path)
