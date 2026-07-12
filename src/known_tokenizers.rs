//! Registry for frozen tokenizer fast paths.
//!
//! This follows the same broad idea as Frokenizer ("Frozen Tokenizer"): identify
//! popular tokenizer configurations and route them through precompiled,
//! tokenizer-specific Rust paths while preserving the generic `tokenizer.json`
//! fallback.

use serde_json::Value;

use crate::{
    json_structs::{PreTokenizerConfig, TokenizerJson},
    pre_tokenizers::QWEN3_SPLIT_PATTERN,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "hf-hub"), allow(dead_code))]
pub(crate) enum KnownTokenizer {
    Qwen3,
    KimiK25,
    Llama3,
}

impl KnownTokenizer {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::KimiK25 => "kimi-k2.5",
            Self::Llama3 => "llama3",
        }
    }

    #[cfg(feature = "known-tokenizer-aliases")]
    pub(crate) fn canonical_model_id(self) -> &'static str {
        match self {
            Self::Qwen3 => "Qwen/Qwen3-0.6B",
            Self::KimiK25 => "hoangquan456/Kimi-K2.5",
            Self::Llama3 => "meta-llama/Meta-Llama-3-8B",
        }
    }
}

#[cfg_attr(not(feature = "hf-hub"), allow(dead_code))]
pub(crate) fn from_model_id(model: &str) -> Option<KnownTokenizer> {
    match model {
        "Qwen/Qwen3-0.6B"
        | "Qwen/Qwen3-235B-A22B-Instruct-2507"
        | "Qwen/Qwen3-Coder-480B-A35B-Instruct"
        | "Qwen/Qwen3-Next-80B-A3B-Thinking"
        | "Qwen/Qwen3-Next-80B-A3B-Instruct"
        | "Qwen/Qwen3.5-397B-A17B"
        | "nvidia/Qwen3-Nemotron-235B-A22B-GenRM" => Some(KnownTokenizer::Qwen3),
        "hoangquan456/Kimi-K2.5" => Some(KnownTokenizer::KimiK25),
        "meta-llama/Meta-Llama-3-8B"
        | "meta-llama/Meta-Llama-3-8B-Instruct"
        | "meta-llama/Meta-Llama-3-70B"
        | "meta-llama/Meta-Llama-3-70B-Instruct"
        | "meta-llama/Meta-Llama-3.1-8B"
        | "meta-llama/Meta-Llama-3.1-8B-Instruct"
        | "meta-llama/Meta-Llama-3.1-70B"
        | "meta-llama/Meta-Llama-3.1-70B-Instruct"
        | "meta-llama/Meta-Llama-3.1-405B"
        | "meta-llama/Meta-Llama-3.1-405B-Instruct"
        | "meta-llama/Llama-3.2-1B"
        | "meta-llama/Llama-3.2-1B-Instruct"
        | "meta-llama/Llama-3.2-3B"
        | "meta-llama/Llama-3.2-3B-Instruct" => Some(KnownTokenizer::Llama3),
        _ => None,
    }
}

#[cfg(feature = "known-tokenizer-aliases")]
#[cfg_attr(not(feature = "hf-hub"), allow(dead_code))]
pub(crate) fn canonical_model_id(model: &str) -> Option<&'static str> {
    from_model_id(model).map(KnownTokenizer::canonical_model_id)
}

pub(crate) fn fingerprint(json: &TokenizerJson) -> Option<KnownTokenizer> {
    if has_qwen3_split_pattern(json) {
        return Some(KnownTokenizer::Qwen3);
    }
    None
}

#[cfg_attr(not(feature = "hf-hub"), allow(dead_code))]
pub(crate) fn vendored_tokenizer_json(model: &str) -> Option<&'static str> {
    from_model_id(model)?;
    // Placeholder for tokenizer JSON blobs once redistribution/license checks
    // have been completed for each known tokenizer family.
    None
}

fn has_qwen3_split_pattern(json: &TokenizerJson) -> bool {
    let Some(pre_tokenizer) = &json.pre_tokenizer else {
        return false;
    };
    pre_tokenizer_contains_qwen3_split(pre_tokenizer)
}

fn pre_tokenizer_contains_qwen3_split(config: &PreTokenizerConfig) -> bool {
    match config {
        PreTokenizerConfig::Split(split) => split.source() == QWEN3_SPLIT_PATTERN,
        PreTokenizerConfig::Sequence { pretokenizers } => {
            pretokenizers.iter().any(pre_tokenizer_contains_qwen3_split)
        }
        PreTokenizerConfig::Other(value) => json_value_contains_qwen3_split(value),
        _ => false,
    }
}

fn json_value_contains_qwen3_split(value: &Value) -> bool {
    match value {
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("Split")
                && let Some(pattern) = map.get("pattern")
                && pattern
                    .get("Regex")
                    .and_then(Value::as_str)
                    .is_some_and(|regex| regex == QWEN3_SPLIT_PATTERN)
            {
                return true;
            }
            map.values().any(json_value_contains_qwen3_split)
        }
        Value::Array(values) => values.iter().any(json_value_contains_qwen3_split),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn qwen_model_aliases_are_known() {
        assert_eq!(
            from_model_id("Qwen/Qwen3-0.6B"),
            Some(KnownTokenizer::Qwen3)
        );
        assert_eq!(
            from_model_id("nvidia/Qwen3-Nemotron-235B-A22B-GenRM"),
            Some(KnownTokenizer::Qwen3),
        );
        assert_eq!(
            from_model_id("hoangquan456/Kimi-K2.5"),
            Some(KnownTokenizer::KimiK25),
        );
        assert_eq!(
            from_model_id("meta-llama/Meta-Llama-3.1-70B-Instruct"),
            Some(KnownTokenizer::Llama3),
        );
        assert_eq!(from_model_id("unknown/model"), None);
    }

    #[test]
    #[cfg(feature = "known-tokenizer-aliases")]
    fn aliases_route_to_canonical_model_ids() {
        assert_eq!(
            canonical_model_id("Qwen/Qwen3-Next-80B-A3B-Instruct"),
            Some("Qwen/Qwen3-0.6B"),
        );
        assert_eq!(
            canonical_model_id("meta-llama/Meta-Llama-3.1-70B-Instruct"),
            Some("meta-llama/Meta-Llama-3-8B"),
        );
        assert_eq!(canonical_model_id("unknown/model"), None);
    }

    #[test]
    fn qwen_fingerprint_detects_split_pattern() {
        let json: TokenizerJson = serde_json::from_value(json!({
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": QWEN3_SPLIT_PATTERN},
                        "behavior": "Isolated",
                        "invert": false
                    }
                ]
            },
            "model": {
                "type": "WordLevel",
                "vocab": {},
                "unk_token": "[UNK]"
            }
        }))
        .unwrap();

        assert_eq!(fingerprint(&json), Some(KnownTokenizer::Qwen3));
    }
}
