# 🧠 Understanding FAISS & RAG — From First Principles

This document explains how our Retrieval-Augmented Generation (RAG) system works — starting from simple vector math and building up to a production-ready backend using FAISS and OpenAI models.

---

## 📌 1. The Core Idea

Modern RAG systems work in three stages:
1. Convert text into vectors (embeddings)
2. Search those vectors using similarity (FAISS)
3. Generate answers using retrieved context (LLM)

At its core:

> We convert language into geometry, search in high-dimensional space, and convert geometry back into language.

---

## 🔢 2. What Is an Embedding?

An embedding is a numeric representation of text.

Example:

`"Dopamine regulates reward."`

Becomes:

`[0.012, -0.88, 0.44, ..., 0.91]`  ← 1536 numbers

If we embed 5 documents and each embedding has 384 dimensions:

`Matrix shape = (5, 384)`

Meaning:
- 5 rows → 5 documents
- 384 columns → semantic features

Each document becomes a point in high-dimensional space.

---

## 📐 3. How Similarity Is Measured

When a user asks a question:

`"What regulates mood?"`

It is also converted into a vector `q`.

To compare it with a stored document vector `xᵢ`, we compute:

`||q - xᵢ||² = Σ (qⱼ - xᵢⱼ)²`

This is L2 (Euclidean) distance.

Vector subtraction happens element-by-element:

`q - xᵢ = [q₁ - xᵢ₁, q₂ - xᵢ₂, ..., q₁₅₃₆ - xᵢ₁₅₃₆]`

Small distance → semantically similar  
Large distance → semantically different

---

## 🚀 4. What FAISS Does

Without FAISS, we would:

```text
For each stored vector:
    compute distance to query
```

That works for 5 vectors — not 500,000.

FAISS provides:
- Fast nearest-neighbor search
- Efficient memory storage
- Optimized C++ performance

When we call:

```python
distances, indices = index.search(q, k)
```

It returns:
- `indices` → positions of the `k` closest vectors
- `distances` → their similarity scores

Important:

**FAISS stores only vectors, not text.**

The indices are just row numbers in the stored embedding matrix.

---

## 🔍 5. What Does k Mean?

`k` is the number of nearest neighbors to retrieve.

If:

```python
index.search(q, 3)
```

FAISS returns the 3 closest vectors.

In our backend:

```text
RETRIEVAL_POOL_K = 24
TOP_K = 8
```

We:
1. Retrieve 24 candidates
2. Re-rank them
3. Keep the best 8

---

## 🧮 6. Why indices[0]?

FAISS supports batch search.

If we search one query vector shaped:

`(1, 1536)`

Results are shaped:

`(1, k)`

So:

`indices[0]`

Gives the `k` neighbors for that one query.

---

## 🔄 7. Hybrid Re-Ranking (Semantic + Keyword)

Pure semantic search sometimes retrieves loosely related chunks.

To improve precision, we:
1. Extract meaningful keywords from the query.
2. Count how many appear in each retrieved chunk.
3. Sort by:
   - Higher keyword matches first
   - Then lower semantic distance

Sorting rule:

```python
retrieved_items.sort(
    key=lambda item: (-item["keyword_hits"], item["distance"])
)
```

This combines:
- Geometry (semantic similarity)
- Symbolic logic (literal keyword match)

This is called **Hybrid Retrieval**.

---

## 🛡 8. Strict RAG Design

After retrieval:
1. We send only the retrieved chunks to the language model.
2. The model must answer strictly from provided context.
3. If no evidence exists → return `NOT_FOUND`.

This prevents hallucination.

There is no fallback to general knowledge.

---

## 🏗 9. Full System Flow

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS Index (vector.index)

User Question
   ↓
Embedding
   ↓
FAISS Search (k nearest)
   ↓
Hybrid Re-rank
   ↓
Top Chunks
   ↓
LLM (strict context answering)
   ↓
Final Answer + Confidence
```

---

## 🎯 10. What We Built

This system:
- Turns language into vectors
- Uses high-dimensional geometry for search
- Applies keyword refinement
- Controls hallucination
- Returns structured confidence scores

At its heart:

> Meaning becomes math.  
> Math retrieves context.  
> Context becomes answer.

---

## 📚 Key Concepts Learned

- Embeddings are coordinates in semantic space
- Vector subtraction measures semantic difference
- FAISS performs nearest-neighbor search
- `indices` are row positions, not vectors
- `k` controls retrieval breadth
- Hybrid search improves accuracy
- Strict RAG reduces hallucination

---

## 🧠 Final Insight

A RAG system is not “just a chatbot.”

It is a geometric retrieval engine wrapped around a language model.

We search meaning in high-dimensional space — and generate answers grounded in retrieved evidence.
