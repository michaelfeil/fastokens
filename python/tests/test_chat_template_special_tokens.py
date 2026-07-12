from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from fastokens import StructuralTokenConfig, Tokenizer

tokenizers = pytest.importorskip("tokenizers")

VENDORED_DIR = Path(__file__).parents[2] / "vendored_tokenizers"


@dataclass(frozen=True)
class ChatTemplateCase:
    name: str
    tokenizer_dir: str
    rendered_prompt: str
    structural_tokens: tuple[str, ...]
    non_special_structural_tokens: tuple[str, ...] = ()


CASES = [
    ChatTemplateCase(
        name="qwen3_tools_system_assistant",
        tokenizer_dir="qwen3-8b",
        rendered_prompt=(
            "<|im_start|>system\n"
            "You are a concise weather assistant. User details are [REDACTED]."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Use the weather tool for [CITY] and answer in Celsius."
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>Need current weather before answering.</think>\n"
            "<tool_call>\n"
            '{"name":"get_weather","arguments":{"city":"[CITY]","unit":"celsius"}}\n'
            "</tool_call><|im_end|>\n"
            "<|im_start|>tool\n"
            '<tool_response>{"temperature":21,"condition":"rain"}</tool_response>'
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "It is 21°C and raining in [CITY].<|im_end|>\n"
        ),
        structural_tokens=(
            "<|im_start|>",
            "<|im_end|>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<think>",
            "</think>",
        ),
        non_special_structural_tokens=(
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<think>",
            "</think>",
        ),
    ),
    ChatTemplateCase(
        name="kimi_k2_5_tools_system_assistant",
        tokenizer_dir="kimi-k2.5",
        rendered_prompt=(
            "[BOS]<|im_system|>\n"
            "You route requests to tools. Private fields are [REDACTED]."
            "<|im_end|>\n"
            "<|im_user|>\n"
            "Find flights from [ORIGIN] to [DESTINATION] and explain the result."
            "<|im_end|>\n"
            "<|im_assistant|>\n"
            "<think>Call the flight search tool, then summarize.</think>\n"
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>flight_search"
            "<|tool_call_argument_begin|>"
            '{"origin":"[ORIGIN]","destination":"[DESTINATION]","date":"[DATE]"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
            "<|im_end|>\n"
            "<|im_assistant|>\n"
            "I found two matching flights for [DATE].<|im_end|>\n"
        ),
        structural_tokens=(
            "[BOS]",
            "<|im_system|>",
            "<|im_user|>",
            "<|im_assistant|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<|tool_calls_section_begin|>",
            "<|tool_calls_section_end|>",
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_end|>",
        ),
    ),
    ChatTemplateCase(
        name="gpt_oss_tools_system_assistant",
        tokenizer_dir="gpt-oss-120b",
        rendered_prompt=(
            "<|startoftext|>"
            "<|start|>system<|message|>"
            "You are a safe coding assistant. Secrets are [REDACTED]."
            "<|end|>"
            "<|start|>user<|message|>"
            "Create a JSON patch for [REDACTED_FILE] and use the checker tool."
            "<|end|>"
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Need to validate the patch before final answer."
            "<|end|>"
            "<|start|>assistant to=checker.run<|channel|>commentary<|message|>"
            '{"path":"[REDACTED_FILE]","mode":"dry_run"}'
            "<|call|>"
            "<|start|>tool checker.run<|message|>"
            '{"ok":true,"warnings":[]}'
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "The dry run succeeded.<|end|>"
        ),
        structural_tokens=(
            "<|startoftext|>",
            "<|start|>",
            "<|message|>",
            "<|end|>",
            "<|channel|>",
            "<|call|>",
        ),
    ),
    ChatTemplateCase(
        name="glm_5_2_tools_system_assistant",
        tokenizer_dir="glm-5.2",
        rendered_prompt=(
            "[gMASK]<sop><|system|>\n"
            "You are GLM with tool access. PII is [REDACTED]."
            "<|user|>\n"
            "Search docs for [QUERY] and cite the tool response."
            "<|assistant|>\n"
            "<think>Use the docs tool and then answer.</think>\n"
            "<tool_call><arg_key>query</arg_key><arg_value>[QUERY]</arg_value>"
            "</tool_call>\n"
            "<|observation|>\n"
            "<tool_response>Result: [REDACTED_DOC_SNIPPET]</tool_response>"
            "<|assistant|>\n"
            "The docs say [REDACTED_DOC_SNIPPET].<eop>"
        ),
        structural_tokens=(
            "[gMASK]",
            "<sop>",
            "<eop>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|observation|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<arg_key>",
            "</arg_key>",
            "<arg_value>",
            "</arg_value>",
        ),
        non_special_structural_tokens=(
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<arg_key>",
            "</arg_key>",
            "<arg_value>",
            "</arg_value>",
        ),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_censored_chat_template_special_tokens_match_tokenizers(
    case: ChatTemplateCase,
) -> None:
    tokenizer_path = VENDORED_DIR / case.tokenizer_dir / "tokenizer.json"
    reference = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    fast = Tokenizer.from_file(str(tokenizer_path))

    expected_ids = reference.encode(
        case.rendered_prompt,
        add_special_tokens=False,
    ).ids
    actual_ids = fast.encode(
        case.rendered_prompt,
        add_special_tokens=False,
    ).ids

    assert actual_ids == expected_ids
    for token in case.structural_tokens:
        token_id = reference.token_to_id(token)
        assert token_id is not None, f"{case.name}: missing token {token!r}"
        assert token_id in actual_ids, f"{case.name}: {token!r} was not emitted"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_structural_chat_template_encoding_matches_tokenizers(
    case: ChatTemplateCase,
) -> None:
    tokenizer_path = VENDORED_DIR / case.tokenizer_dir / "tokenizer.json"
    reference = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    fast = Tokenizer.from_file(str(tokenizer_path))
    structural_config = StructuralTokenConfig(
        case.structural_tokens,
        case.non_special_structural_tokens,
    )

    expected_ids = reference.encode(
        case.rendered_prompt,
        add_special_tokens=False,
    ).ids
    actual_ids = fast.encode_with_structural_tokens(
        case.rendered_prompt,
        structural_config,
    ).ids

    assert actual_ids == expected_ids
