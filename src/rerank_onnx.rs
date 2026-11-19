//! ONNX Runtime CPU reranker backend.
//!
//! The model and tokenizer are external artifacts. nVDB owns the ranking
//! orchestration; this module only turns query/document pairs into scores.

use std::path::PathBuf;
use std::sync::Mutex;

use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::error::{Error, Result};
use crate::rerank::Reranker;

/// ONNX reranker configuration.
#[derive(Debug, Clone)]
pub struct OnnxRerankerConfig {
    /// Path to the sequence-classification ONNX model.
    pub model_path: PathBuf,
    /// Path to the Hugging Face `tokenizer.json`.
    pub tokenizer_path: PathBuf,
    /// Maximum tokens per query/document pair.
    pub max_length: usize,
    /// Pairs per inference call.
    pub batch_size: usize,
    /// Optional ONNX Runtime intra-op thread count.
    pub intra_threads: Option<usize>,
}

impl OnnxRerankerConfig {
    /// Create a configuration with the measured default pair budget.
    pub fn new(model_path: impl Into<PathBuf>, tokenizer_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tokenizer_path: tokenizer_path.into(),
            max_length: 112,
            batch_size: 10,
            intra_threads: None,
        }
    }
}

/// CPU ONNX cross-encoder.
pub struct OnnxReranker {
    _environment: Environment,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    batch_size: usize,
}

impl OnnxReranker {
    /// Load a tokenizer and ONNX session.
    pub fn load(config: OnnxRerankerConfig) -> Result<Self> {
        if config.max_length == 0 || config.batch_size == 0 {
            return Err(Error::invalid_arg(
                "reranker",
                "max_length and batch_size must be positive",
            ));
        }
        if !config.model_path.is_file() {
            return Err(Error::invalid_arg(
                "model_path",
                format!("ONNX model not found: {}", config.model_path.display()),
            ));
        }
        if !config.tokenizer_path.is_file() {
            return Err(Error::invalid_arg(
                "tokenizer_path",
                format!("tokenizer not found: {}", config.tokenizer_path.display()),
            ));
        }

        let mut tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|error| Error::invalid_arg("tokenizer_path", error.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_length,
                ..TruncationParams::default()
            }))
            .map_err(|error| Error::invalid_arg("tokenizer", error.to_string()))?;
        let pad_id = tokenizer
            .token_to_id("<pad>")
            .ok_or_else(|| Error::invalid_arg("tokenizer", "tokenizer has no <pad> token"))?;
        tokenizer.with_padding(Some(PaddingParams {
            pad_id,
            pad_token: "<pad>".to_string(),
            ..PaddingParams::default()
        }));

        let environment = ort::init()
            .with_name("nvdb-rerank")
            .build()
            .map_err(|error| Error::invalid_arg("onnx", error.to_string()))?;
        let mut builder = Session::builder(&environment)
            .map_err(|error| Error::invalid_arg("onnx", error.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|error| Error::invalid_arg("onnx", error.to_string()))?;
        if let Some(threads) = config.intra_threads {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|error| Error::invalid_arg("onnx", error.to_string()))?;
        }
        let session = builder
            .commit_from_file(&config.model_path)
            .map_err(|error| Error::invalid_arg("onnx", error.to_string()))?;

        Ok(Self {
            _environment: environment,
            session: Mutex::new(session),
            tokenizer,
            batch_size: config.batch_size,
        })
    }
}

impl Reranker for OnnxReranker {
    fn score(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(documents.len());
        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::invalid_arg("reranker", "ONNX session lock poisoned"))?;

        for batch in documents.chunks(self.batch_size) {
            let pairs: Vec<(&str, &str)> =
                batch.iter().map(|document| (query, *document)).collect();
            let encodings = self
                .tokenizer
                .encode_batch(pairs, true)
                .map_err(|error| Error::invalid_arg("reranker", error.to_string()))?;
            if encodings.is_empty() {
                continue;
            }
            let sequence_length = encodings[0].get_ids().len();
            let mut input_ids = Vec::with_capacity(batch.len() * sequence_length);
            let mut attention_mask = Vec::with_capacity(batch.len() * sequence_length);
            for encoding in &encodings {
                if encoding.get_ids().len() != sequence_length {
                    return Err(Error::invalid_arg(
                        "reranker",
                        "tokenizer padding produced inconsistent sequence lengths",
                    ));
                }
                input_ids.extend(encoding.get_ids().iter().map(|id| i64::from(*id)));
                attention_mask.extend(
                    encoding
                        .get_attention_mask()
                        .iter()
                        .map(|mask| i64::from(*mask)),
                );
            }

            let ids = Tensor::from_array((vec![batch.len(), sequence_length], input_ids))
                .map_err(|error| Error::invalid_arg("onnx", error.to_string()))?;
            let mask = Tensor::from_array((vec![batch.len(), sequence_length], attention_mask))
                .map_err(|error| Error::invalid_arg("onnx", error.to_string()))?;
            let outputs = session
                .run(ort::inputs!["input_ids" => ids, "attention_mask" => mask])
                .map_err(|error| Error::invalid_arg("onnx", error.to_string()))?;
            let logits = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|error| Error::invalid_arg("onnx", error.to_string()))?
                .1;
            if logits.len() != batch.len() {
                return Err(Error::invalid_arg(
                    "onnx",
                    format!(
                        "expected {} reranker logits, got {}",
                        batch.len(),
                        logits.len()
                    ),
                ));
            }
            for logit in logits {
                scores.push(1.0 / (1.0 + (-logit).exp()));
            }
        }
        Ok(scores)
    }
}
