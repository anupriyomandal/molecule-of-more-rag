# Search Algorithm Changes and Learnings

## Why We Needed to Improve Retrieval + Answering
The app is designed as **strict RAG**: answers must come only from retrieved book context. During testing, we saw two failure patterns:
- Relevant passages were retrieved, but model still returned `NOT_FOUND`.
- Some top passages matched query words weakly or indirectly, causing lower answer quality.

We iteratively improved both retrieval and decision logic.

---

## 1) Removed GPT Fallback (Strict RAG Enforcement)
### Original behavior
- If RAG answer was weak or `NOT_FOUND`, the app fell back to a generic GPT answer.

### Change made
- Removed fallback path completely.
- If unsupported by context, API returns explicit `RAG_NO_ANSWER`.

### Learning
- This improved trust and source-grounding.
- It also surfaced retrieval weaknesses that fallback had hidden.

---

## 2) Distance Threshold Tuning
### Original behavior
- Very strict threshold (`DISTANCE_THRESHOLD = 1.2`) triggered false no-answer responses.

### Change made
- Increased threshold to `1.5` to reduce over-rejection.

### Learning
- Overly strict thresholds hurt recall.
- Thresholds must be calibrated to embedding model + chunk strategy.

---

## 3) Better No-Answer Decision Logic
### Original behavior
- No-answer triggered too aggressively from first-pass model response.

### Change made
- Added exact sentinel check (`is_not_found`) for robust `NOT_FOUND` handling.
- Prevented raw `NOT_FOUND` leakage to UI.
- Added second strict context-only retry before returning no-answer.

### Learning
- A single strict pass can be too brittle.
- Retry logic can reduce false negatives while still staying strict RAG.

---

## 4) Retrieval Pool + Hybrid Reranking
### Original behavior
- Retrieval relied mainly on semantic nearest neighbors.

### Change made
- Increased candidate pool (`RETRIEVAL_POOL_K = 24`).
- Added lexical signal (`keyword_hits`) from query terms.
- Reranked by `keyword_hits` first, then vector distance.
- Kept top `TOP_K` chunks for final context.

### Learning
- Hybrid retrieval (semantic + lexical) outperforms semantic-only for definition and keyword-sensitive questions.
- Larger initial pool improves recall before final pruning.

---

## 5) Explainability and Debug Signals in API/UI
### Change made
- API now returns:
  - `confidence`
  - `sources_used`
  - `top_passages`
  - per-passage `distance` and `confidence`
- Frontend shows top 5 passages with query-term highlights.

### Learning
- Visibility into evidence makes debugging much faster.
- Users can validate answers instead of trusting black-box outputs.

---

## 6) Re-ingestion and Index Sync
### Change made
- Re-ran `ingest.py` and updated retrieval artifacts (`vector.index`).

### Learning
- Retrieval quality depends on fresh, synchronized artifacts.
- `vector.index` and `chunks.npy` must come from the same ingestion run.

---

## Final Algorithm (Current)
1. Embed user query.
2. Search FAISS for `RETRIEVAL_POOL_K` nearest chunks.
3. Compute lexical overlap (`keyword_hits`) for each candidate.
4. Rerank candidates by lexical hits + semantic distance.
5. Select top `TOP_K` chunks as prompt context.
6. Generate strict context-grounded answer.
7. If model returns `NOT_FOUND`, retry once with strongest lexical passages (still context-only).
8. If still `NOT_FOUND`, return `RAG_NO_ANSWER`.
9. Return answer + confidence + top evidence passages.

---

## Key Takeaways for Presentation
- **Strict RAG is a policy choice**: high trust, lower hallucination risk.
- **Retrieval quality drives answer quality** more than model choice alone.
- **Hybrid reranking and calibrated thresholds** are essential.
- **Explainability (passages/confidence)** is not optional in educational/document QA systems.
