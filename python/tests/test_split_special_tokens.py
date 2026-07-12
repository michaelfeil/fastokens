import json
import unittest
from copy import deepcopy

import numpy as np

from fastokens._compat import _TokenizerShim
from fastokens._native import Encoding, StructuralTokenConfig, Tokenizer


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
        (
            "ascii-endoftext",
            "<|endoftext|>",
            "<tool>",
        ),
        (
            "ascii-short",
            "<s>",
            "<tool_call>",
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

    def test_native_split_special_tokens_handles_specials_inside_text(self) -> None:
        for model_name, special_token, _non_special_token in self.CASES:
            with self.subTest(model_name=model_name):
                tokenizer_json = _tokenizer_json(special_token, "<tool>")
                tokenizer = Tokenizer.from_json_str(tokenizer_json)
                text = f"hello{special_token} hello"

                default_ids = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                ).ids
                split_ids = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    split_special_tokens=True,
                ).ids

                self.assertIn(SPECIAL_ID, default_ids)
                self.assertEqual(split_ids, _char_ids(tokenizer_json, text))

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

    def test_shim_respects_encode_special_tokens_property(self) -> None:
        for model_name, special_token, _non_special_token in self.CASES:
            with self.subTest(model_name=model_name):
                tokenizer_json = _tokenizer_json(special_token, "<tool>")
                tokenizer = _TokenizerShim.from_str(tokenizer_json)
                tokenizer.encode_special_tokens = True

                self.assertEqual(
                    tokenizer.encode(special_token, add_special_tokens=False).ids,
                    _char_ids(tokenizer_json, special_token),
                )
                self.assertEqual(
                    tokenizer.encode_batch(
                        [special_token],
                        add_special_tokens=False,
                    )[0].ids,
                    _char_ids(tokenizer_json, special_token),
                )

    def test_shim_explicit_split_special_tokens_overrides_property(self) -> None:
        for model_name, special_token, _non_special_token in self.CASES:
            with self.subTest(model_name=model_name):
                tokenizer_json = _tokenizer_json(special_token, "<tool>")
                tokenizer = _TokenizerShim.from_str(tokenizer_json)
                tokenizer.encode_special_tokens = True

                self.assertEqual(
                    tokenizer.encode(
                        special_token,
                        add_special_tokens=False,
                        split_special_tokens=False,
                    ).ids,
                    [SPECIAL_ID],
                )

    def test_shim_copy_preserves_encode_special_tokens_property(self) -> None:
        for model_name, special_token, _non_special_token in self.CASES:
            with self.subTest(model_name=model_name):
                tokenizer_json = _tokenizer_json(special_token, "<tool>")
                tokenizer = _TokenizerShim.from_str(tokenizer_json)
                tokenizer.encode_special_tokens = True

                wrapped = _TokenizerShim(tokenizer)
                copied = deepcopy(tokenizer)

                self.assertTrue(wrapped.encode_special_tokens)
                self.assertEqual(
                    wrapped.encode(special_token, add_special_tokens=False).ids,
                    _char_ids(tokenizer_json, special_token),
                )
                self.assertTrue(copied.encode_special_tokens)
                self.assertEqual(
                    copied.encode(special_token, add_special_tokens=False).ids,
                    _char_ids(tokenizer_json, special_token),
                )

    def test_native_encode_with_structural_tokens(self) -> None:
        tokenizer_json = _tokenizer_json("<think>", "<tool>")
        tokenizer = Tokenizer.from_json_str(tokenizer_json)
        think_placeholder = "\ue000STRUCTTOK_0\ue000"
        tool_placeholder = "\ue000STRUCTTOK_1\ue000"
        structural_config = StructuralTokenConfig(
            {"<think>", "<tool>"},
            {"<tool>"},
        )

        ids = tokenizer.encode_with_structural_tokens(
            f"hello <think> {think_placeholder} <tool> {tool_placeholder}",
            structural_config,
            {
                think_placeholder: "<think>",
                tool_placeholder: "<tool>",
            },
        ).ids

        self.assertEqual(
            ids,
            [
                *_char_ids(tokenizer_json, "hello "),
                SPECIAL_ID,
                *_char_ids(tokenizer_json, " "),
                *_char_ids(tokenizer_json, "<think>"),
                *_char_ids(tokenizer_json, " "),
                NON_SPECIAL_ID,
                *_char_ids(tokenizer_json, " "),
                *_char_ids(tokenizer_json, "<tool>"),
            ],
        )

    def test_encoding_into_numpy_moves_selected_fields(self) -> None:
        encoding = Encoding([1, 2, 3])

        arrays = encoding.into_numpy(
            attention_mask=True,
            type_ids=True,
            special_tokens_mask=True,
        )

        self.assertEqual(
            set(arrays),
            {"ids", "attention_mask", "type_ids", "special_tokens_mask"},
        )
        ids = arrays["ids"]
        self.assertEqual(ids.dtype, np.dtype("uint32"))
        np.testing.assert_array_equal(ids, np.array([1, 2, 3], dtype=np.uint32))
        np.testing.assert_array_equal(
            arrays["attention_mask"],
            np.array([1, 1, 1], dtype=np.uint32),
        )
        np.testing.assert_array_equal(
            arrays["type_ids"],
            np.array([0, 0, 0], dtype=np.uint32),
        )
        np.testing.assert_array_equal(
            arrays["special_tokens_mask"],
            np.array([0, 0, 0], dtype=np.uint32),
        )
        self.assertEqual(encoding.ids, [])
        self.assertEqual(encoding.attention_mask, [])
        self.assertEqual(encoding.type_ids, [])
        self.assertEqual(encoding.special_tokens_mask, [])
        self.assertEqual(encoding.n_sequences, 0)

    def test_encoding_into_numpy_can_skip_ids(self) -> None:
        encoding = Encoding([1, 2, 3])

        arrays = encoding.into_numpy(ids=False, attention_mask=True)

        self.assertEqual(set(arrays), {"attention_mask"})
        np.testing.assert_array_equal(
            arrays["attention_mask"],
            np.array([1, 1, 1], dtype=np.uint32),
        )
        self.assertEqual(encoding.ids, [])
        self.assertEqual(encoding.attention_mask, [])

    def test_encoding_into_numpy_rejects_empty_selection(self) -> None:
        encoding = Encoding([1, 2, 3])

        with self.assertRaisesRegex(ValueError, "at least one field"):
            encoding.into_numpy(
                ids=False,
                attention_mask=False,
                type_ids=False,
                special_tokens_mask=False,
            )

    def test_native_encode_with_structural_tokens_can_add_special_tokens(self) -> None:
        tokenizer_json = _tokenizer_json("<think>", "<tool>")
        tokenizer = Tokenizer.from_json_str(tokenizer_json)
        structural_config = StructuralTokenConfig({"<think>"})

        ids = tokenizer.encode_with_structural_tokens(
            "hello <think>",
            structural_config,
            add_special_tokens=True,
        ).ids

        self.assertEqual(
            ids,
            [BOS_ID, *_char_ids(tokenizer_json, "hello "), SPECIAL_ID],
        )

    def test_structural_token_config_rejects_single_string(self) -> None:
        with self.assertRaisesRegex(TypeError, "not a single string"):
            StructuralTokenConfig("<think>")

        with self.assertRaisesRegex(TypeError, "not a single string"):
            StructuralTokenConfig({"<think>"}, "<tool>")

    def test_native_encode_with_structural_tokens_keeps_bare_non_special_added_tokens(
        self,
    ) -> None:
        tokenizer_json = _tokenizer_json("<think>", "magic")
        tokenizer = Tokenizer.from_json_str(tokenizer_json)
        structural_config = StructuralTokenConfig({"<think>"})

        ids = tokenizer.encode_with_structural_tokens(
            "hello magic <think>",
            structural_config,
        ).ids

        self.assertEqual(
            ids,
            [
                *_char_ids(tokenizer_json, "hello "),
                NON_SPECIAL_ID,
                *_char_ids(tokenizer_json, " "),
                SPECIAL_ID,
            ],
        )

    def test_native_encode_with_structural_tokens_ignores_padding_and_truncation(
        self,
    ) -> None:
        tokenizer_json = _tokenizer_json("<think>", "<tool>")
        tokenizer = Tokenizer.from_json_str(tokenizer_json)
        tokenizer.enable_truncation(max_length=2)
        tokenizer.enable_padding(length=8, pad_id=99)
        structural_config = StructuralTokenConfig({"<think>"})

        encoding = tokenizer.encode_with_structural_tokens(
            "hello <think>",
            structural_config,
        )

        self.assertEqual(
            encoding.ids,
            [*_char_ids(tokenizer_json, "hello "), SPECIAL_ID],
        )
        self.assertEqual(encoding.attention_mask, [1] * len(encoding.ids))

    def test_shim_forwards_encode_with_structural_tokens(self) -> None:
        tokenizer_json = _tokenizer_json("<think>", "<tool>")
        tokenizer = _TokenizerShim.from_str(tokenizer_json)
        think_placeholder = "\ue000STRUCTTOK_0\ue000"
        structural_config = StructuralTokenConfig({"<think>"})

        ids = tokenizer.encode_with_structural_tokens(
            f"<think>{think_placeholder}",
            structural_config,
            {think_placeholder: "<think>"},
        ).ids

        self.assertEqual(
            ids,
            [SPECIAL_ID, *_char_ids(tokenizer_json, "<think>")],
        )

    def test_shim_rejects_unknown_encode_kwargs(self) -> None:
        tokenizer = _TokenizerShim.from_str(_tokenizer_json("<think>", "<tool>"))

        with self.assertRaisesRegex(TypeError, "return_offsets_mapping"):
            tokenizer.encode(
                "hello",
                add_special_tokens=False,
                return_offsets_mapping=True,
            )

        with self.assertRaisesRegex(TypeError, "return_attention_mask"):
            tokenizer.encode_batch(
                ["hello"],
                add_special_tokens=False,
                return_attention_mask=True,
            )


if __name__ == "__main__":
    unittest.main()
