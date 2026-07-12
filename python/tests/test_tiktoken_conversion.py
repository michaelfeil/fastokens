import base64
import builtins
import gzip
import json
from pathlib import Path

import pytest

from fastokens import Tokenizer, tiktoken_model_to_tokenizer_json, tiktoken_to_tokenizer_json

VENDORED_DIR = Path(__file__).parents[2] / "vendored_tokenizers"


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


def _write_tiktoken_model(path: Path, encoding) -> None:
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
    pytest.importorskip("tiktoken")
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


def test_kimi_k2_5_tiktoken_gz_conversion_matches_vendored_tokenizer_json(tmp_path) -> None:
    """Convert the vendored Kimi K2.5 tiktoken.model.gz on the fly and verify
    that encoding results match the vendored tokenizer.json for a corpus of
    test strings.  Requires tiktoken; skipped otherwise."""
    pytest.importorskip("tiktoken")

    kimi_dir = VENDORED_DIR / "kimi-k2.5"
    gz_path = kimi_dir / "tiktoken.model.gz"
    vendored_json_path = kimi_dir / "tokenizer.json"

    if not gz_path.exists() or not vendored_json_path.exists():
        pytest.skip("vendored kimi-k2.5 assets not found")

    # Decompress the tiktoken model to a temporary plain file.
    model_path = tmp_path / "tiktoken.model"
    model_path.write_bytes(gzip.decompress(gz_path.read_bytes()))

    # Extract the pre-tokenizer pattern and special tokens from the vendored
    # tokenizer.json so the conversion uses exactly the same configuration.
    vendored = json.loads(vendored_json_path.read_text())
    pattern: str = vendored["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    special_tokens: dict[str, int] = {
        t["content"]: t["id"]
        for t in vendored["added_tokens"]
        if t.get("special")
    }

    # Convert the tiktoken model to a tokenizer JSON and build a Tokenizer.
    converted_json = tiktoken_model_to_tokenizer_json(
        model_path,
        pattern=pattern,
        special_tokens=special_tokens,
    )
    assert converted_json is not None, "tiktoken_model_to_tokenizer_json returned None"
    converted_tok = Tokenizer.from_json_str(converted_json)

    # Also build a Tokenizer directly from the vendored tokenizer.json.
    vendored_tok = Tokenizer.from_json_str(vendored_json_path.read_text())

    corpus = [
        "Hello, world!",
        "1 + 2 = 3",
        "The quick brown fox jumps over the lazy dog.",
        "你好，世界",
        "fastokens converts tiktoken models on the fly",
    ]
    for text in corpus:
        converted_ids = converted_tok.encode(text, add_special_tokens=False).ids
        vendored_ids = vendored_tok.encode(text, add_special_tokens=False).ids
        assert converted_ids == vendored_ids, (
            f"encoding mismatch for {text!r}:\n"
            f"  converted : {converted_ids}\n"
            f"  vendored  : {vendored_ids}"
        )
