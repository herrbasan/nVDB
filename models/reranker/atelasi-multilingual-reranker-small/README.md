# atelasi/multilingual-reranker-small

Licence-clean production candidate.

- Source: `atelasi/multilingual-reranker-small`
- Revision: `ba4b00057d7bf2c009ec0a3888cebb8aa87852e0`
- Licence: **Apache-2.0**
- Base model: `jhu-clsp/mmBERT-small`
- Artifact: `onnx/model_quantized.onnx` → local `model.onnx`
- Tokenizer: `tokenizer.json` from the same revision
- Languages: 40+, including German and English

Local SHA-256:

- `model.onnx`:
  `F913154682468BFE92D6A7E02ED4E994E548C112FDDC74452B573B17C949046E`
- `tokenizer.json`:
  `7925F638B5444897EFE3B4608EED56C1652F0ED39C2662C8302DE1BC264C6088`

Measured on the real labeled corpus with misses counted as failures:

- max context: `64`
- hybrid RRF: `k = 10`
- top-10 rerank, rank-fusion weight `0.2`
- MRR: `0.7907 → 0.8148`
- top-1: `112 → 119`
- 17 improvements / 2 regressions
- reranker overhead: approximately 90–110 ms

The 268 MB ONNX file and 34 MB tokenizer are ignored by Git. Verify both hashes
during provisioning.
