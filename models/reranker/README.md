# Reranker model artifacts

The `rerank-onnx` feature expects model files here. The binary artifacts are
ignored by Git and provisioned separately.

## Default candidate

- Model: `Slite/mmarco-mMiniLMv2-L6-H384-v1-onnx-o4`
- File: `model_optimized.onnx` → local `model.onnx`
- Revision: `1347818c9e1c7673672d5cb0bc5ecd6a521e872e`
- Tokenizer: `tokenizer.json` from the same revision
- Intended settings: `max_length = 112`, top-10 shortlist, blend weight `0.2`

Local sizes and SHA-256:

- `model.onnx`: `214017536` bytes
  `0EAAE91D96A80CCD9AF77C8CD0AB7C5B799C8AA9180ED83B8B4528311A812660`
- `tokenizer.json`: `17082756` bytes
  `F59925FCB90C92B894CB93E51BB9B4A6105C5C249FE54CE1C704420AC39B81AF`

## Licence status

The Hugging Face export and its base model
`nreimers/mmarco-mMiniLMv2-L6-H384-v1` do not declare a licence in model
metadata, and the base repository has no model card. **Do not redistribute these
artifacts in a product until the licence is resolved.**

Options:

1. obtain a written licence clarification;
2. retrain/export a clean model from a clearly licensed base;
3. ship an alternative reranker with an explicit licence, such as the
   Apache-2.0 `SugoLabs/mmarco-mMiniLMv2-L12-H384-v1`, accepting its higher
   latency.

The artifacts may remain locally untracked for development and evaluation.

## Provisioning

Place files at:

```text
models/reranker/mmarco-mMiniLMv2-L6-H384-v1/model.onnx
models/reranker/mmarco-mMiniLMv2-L6-H384-v1/tokenizer.json
```

Verify both hashes before use. The ONNX runtime and model paths are explicit
configuration; nVDB does not download models at runtime.
