import pickle

import pytest

from fastokens._compat import _TokenizerShim
from test_async_stub import TOKENIZER_JSON


def test_encode_rejects_split_special_tokens_true() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    with pytest.raises(NotImplementedError, match="split_special_tokens=True"):
        tokenizer.encode("hello", split_special_tokens=True)


def test_encode_batch_rejects_split_special_tokens_true() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    with pytest.raises(NotImplementedError, match="split_special_tokens=True"):
        tokenizer.encode_batch(["hello"], split_special_tokens=True)


def test_encode_accepts_split_special_tokens_false_as_noop() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    plain = tokenizer.encode("hello")
    explicit_false = tokenizer.encode("hello", split_special_tokens=False)

    assert explicit_false.ids == plain.ids


def test_encode_rejects_unknown_kwargs() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    with pytest.raises(TypeError, match="return_offsets_mapping"):
        tokenizer.encode("hello", return_offsets_mapping=True)


def test_encode_special_tokens_true_is_rejected() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    with pytest.raises(NotImplementedError, match="encode_special_tokens=True"):
        tokenizer.encode_special_tokens = True


def test_encode_special_tokens_false_is_noop() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    tokenizer.encode_special_tokens = False

    assert tokenizer.encode_special_tokens is False


def test_pickle_rejects_stored_encode_special_tokens_true() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)
    state = (
        tokenizer.to_str(),
        tokenizer.truncation,
        tokenizer.padding,
        True,
    )

    payload = pickle.dumps(state)
    restored = _TokenizerShim.from_str(TOKENIZER_JSON)

    with pytest.raises(NotImplementedError, match="encode_special_tokens=True"):
        restored.__setstate__(pickle.loads(payload))
