import json
import unittest

from fastokens._compat import _TokenizerShim
from fastokens._native import Tokenizer


BOS_TOKEN = "[BOS]"
BOS_ID = 100
SPECIAL_ID = 101
NON_SPECIAL_ID = 102


def _tokenizer_json(special_token: str, non_special_token: str) -> str:
    chars = sorted(set(BOS_TOKEN + special_token + non_special_token + " hello"))
    vocab = {ch: i for i, ch in enumerate(chars)}
    return json.dumps(
        {
            "version": "1.0",
            "added_tokens": [
                {
                    "id": BOS_ID,
                    "content": BOS_TOKEN,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
                {
                    "id": SPECIAL_ID,
                    "content": special_token,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
                {
                    "id": NON_SPECIAL_ID,
                    "content": non_special_token,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": False,
                },
            ],
            "normalizer": None,
            "pre_tokenizer": None,
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": BOS_TOKEN, "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                ],
                "pair": [],
                "special_tokens": {
                    BOS_TOKEN: {
                        "id": BOS_TOKEN,
                        "ids": [BOS_ID],
                        "tokens": [BOS_TOKEN],
                    }
                },
            },
            "decoder": None,
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": False,
                "byte_fallback": False,
                "vocab": vocab,
                "merges": [],
            },
        }
    )


def _char_ids(tokenizer_json: str, text: str) -> list[int]:
    vocab = json.loads(tokenizer_json)["model"]["vocab"]
    return [vocab[ch] for ch in text]


class SplitSpecialTokensTests(unittest.TestCase):
    CASES = [
        (
            "kimi2.5",
            "<think>",
            "<|tool_calls_section_begin|>",
        ),
    ]

    def test_native_split_special_tokens_matches_kimi_style_expectations(self) -> None:
        for model_name, special_token, non_special_token in self.CASES:
            with self.subTest(model_name=model_name):
                tokenizer_json = _tokenizer_json(special_token, non_special_token)
                tokenizer = Tokenizer.from_json_str(tokenizer_json)
                expected_text_ids = _char_ids(tokenizer_json, special_token)

                default_ids = tokenizer.encode(
                    special_token,
                    add_special_tokens=False,
                ).ids
                split_ids = tokenizer.encode(
                    special_token,
                    add_special_tokens=False,
                    split_special_tokens=True,
                ).ids
                non_special_ids = tokenizer.encode(
                    non_special_token,
                    add_special_tokens=False,
                    split_special_tokens=True,
                ).ids
                with_bos_ids = tokenizer.encode(
                    special_token,
                    add_special_tokens=True,
                    split_special_tokens=True,
                ).ids

                self.assertEqual(default_ids, [SPECIAL_ID])
                self.assertEqual(split_ids, expected_text_ids)
                self.assertEqual(non_special_ids, [NON_SPECIAL_ID])
                self.assertEqual(with_bos_ids, [BOS_ID, *expected_text_ids])

    def test_shim_forwards_split_special_tokens_keyword(self) -> None:
        for model_name, special_token, _non_special_token in self.CASES:
            with self.subTest(model_name=model_name):
                tokenizer_json = _tokenizer_json(special_token, "<tool>")
                tokenizer = _TokenizerShim.from_str(tokenizer_json)

                self.assertEqual(
                    tokenizer.encode(
                        special_token,
                        add_special_tokens=False,
                    ).ids,
                    [SPECIAL_ID],
                )
                self.assertEqual(
                    tokenizer.encode(
                        special_token,
                        add_special_tokens=False,
                        split_special_tokens=True,
                    ).ids,
                    _char_ids(tokenizer_json, special_token),
                )


if __name__ == "__main__":
    unittest.main()
