# Anupriyo Mandal's Ebook Explorer
## Extended Technical Presentation Draft (15 Slides)

---

## Slide 1: Project Overview
### On slide
- **Project:** Anupriyo Mandal's Ebook Explorer
- **Use case:** Ask questions about *The Molecule of More*
- **Approach:** Strict Retrieval-Augmented Generation (RAG)
- **Outcome:** Answers grounded in retrieved book passages

### Speaker notes
I built a strict RAG app around one ebook. Users ask natural-language questions, and answers are generated only from retrieved context from the book.

---

## Slide 2: Problem Definition & Requirements (Detailed)
### On slide
- **Core problem:** Standard LLMs can hallucinate or answer beyond source text
- **Question:** How do I force answers to come from this ebook only?
- **Why it matters:** In learning workflows, source fidelity is critical

**Failure modes to avoid**
- Generic GPT answers not present in the book
- Correct-sounding but unverifiable responses
- No visibility into supporting evidence
- False no-answer despite partial clues

**Functional requirements**
- Ingest PDF and make it searchable
- Retrieve relevant passages for each query
- Answer from context only
- Return no-answer if evidence is absent
- Show top supporting passages

**Non-functional requirements**
- Explainability (confidence + evidence)
- Predictable behavior
- Simple deployment (Railway + Vercel)
- Easy re-ingestion when document changes

### Speaker notes
The challenge is balancing strictness and usability: avoiding hallucinations without over-rejecting valid questions.

---

## Slide 3: End-to-End Architecture
### On slide
- **Offline path:** PDF -> chunking -> embeddings -> FAISS index files
- **Online path:** Query -> embedding -> retrieval -> reranking -> LLM answer
- **Frontend:** Answer + top passages + confidence + highlight

### Speaker notes
I separated ingestion from query-time serving. Ingestion prepares search artifacts once; query pipeline uses them for fast runtime retrieval.

---

## Slide 4: What is RAG in this Project?
### On slide
- **Retrieval:** Find relevant book chunks first
- **Augmentation:** Pass those chunks as context to the model
- **Generation:** Model answers only from that context
- **Strict mode:** No fallback to open-domain GPT answers

### Speaker notes
RAG reduces hallucination risk by forcing evidence-first answering.

---

## Slide 5: What is FAISS ("fassi")?
### On slide
- **FAISS:** Facebook AI Similarity Search library
- Efficient nearest-neighbor search over vectors
- Stores embeddings and returns closest chunks by distance
- Used here via `faiss.IndexFlatL2`

### Speaker notes
FAISS is the retrieval engine. It allows fast semantic search over chunk embeddings.

---

## Slide 6: What is a Vector Index?
### On slide
- Embedding = numeric representation of text meaning
- Vector index = data structure to search similar embeddings
- In this app: `backend/vector.index`
- Query embedding is matched against chunk embeddings

### Speaker notes
The vector index is how semantic similarity search is made practical at runtime.

---

## Slide 7: What is Chunking and Why?
### On slide
- Split long book text into smaller overlapping chunks
- Prevent context overflow in model prompts
- Improve retrieval granularity
- Overlap preserves meaning across chunk boundaries

### Speaker notes
Chunking is essential; without it, retrieval is either too coarse or misses definitions that cross boundaries.

---

## Slide 8: What is Top-K Retrieval?
### On slide
- `K` = number of retrieved chunks kept as candidates
- Retrieve larger pool, then rerank
- Keep best chunks for final context (Top-K)
- Trade-off: higher K = more recall, too high K = noisy context

### Speaker notes
I use semantic retrieval plus reranking so top context is both relevant and keyword-aligned.

---

## Slide 9: How Search Works in This App
### On slide
1. Embed user question
2. FAISS returns nearest chunk candidates
3. Hybrid rerank (keyword hits + distance)
4. Build context from top chunks
5. LLM answers from context
6. If no evidence -> explicit no-answer

### Speaker notes
This avoids pure semantic drift and reduces false negatives.

---

## Slide 10: What is `.npy` and Why Used?
### On slide
- `.npy` = NumPy binary file format
- Stores arrays efficiently on disk
- In this app: `backend/chunks.npy`
- Holds text chunks aligned with vector index positions

### Speaker notes
When FAISS returns index positions, `chunks.npy` lets me map those positions back to actual text passages.

---

## Slide 11: Backend File - `main.py`
### On slide
- FastAPI service and `/ask` endpoint
- Loads `vector.index` and `chunks.npy`
- Performs query embedding + retrieval + reranking
- Calls LLM with strict context-only prompt
- Returns answer, mode, confidence, sources, top passages

### Speaker notes
`main.py` is the live query engine of the application.

---

## Slide 12: Backend File - `ingest.py`
### On slide
- Reads PDF (`../data/document.pdf`)
- Extracts text with PyPDF
- Chunks text with overlap
- Generates embeddings for chunks
- Builds and saves FAISS + `.npy` artifacts

### Speaker notes
`ingest.py` is run whenever source content or chunk/embedding strategy changes.

---

## Slide 13: Backend File - `requirements.txt`
### On slide
- Declares backend dependencies:
  - `fastapi`, `uvicorn`
  - `openai`, `python-dotenv`
  - `faiss-cpu`, `pypdf`, `numpy`
- Ensures reproducible environment for local/deploy

### Speaker notes
This file is used during setup/deploy to install exact packages needed by backend services.

---

## Slide 14: Backend File - `Procfile`
### On slide
- Defines process start command for Railway
- Tells platform how to launch FastAPI/uvicorn app
- Required for consistent production startup behavior

### Speaker notes
Without correct process config, deployment can succeed but service won’t run properly.

---

## Slide 15: Backend Artifacts - `vector.index` and `chunks.npy`
### On slide
- `vector.index`: searchable embedding index (FAISS)
- `chunks.npy`: source text chunk store
- Must stay in sync (same ingestion run)
- If document changes: regenerate both

### Speaker notes
These artifacts are tightly coupled. Mismatched versions can cause poor retrieval quality.

---

## Backup Slide A: Frontend Explainability Features
### On slide
- Displays answer + confidence
- Shows top 5 passages used
- Highlights matched query words
- Helps users verify evidence quickly

---

## Backup Slide B: Challenges and Fixes
### On slide
- Removed GPT fallback to keep strict RAG
- Fixed false no-answer behavior with retry and reranking
- Prevented raw `NOT_FOUND` leakage to UI
- Re-ingested index artifacts after logic updates

---

## Backup Slide C: Suggested Demo Script (2 minutes)
### On slide
1. Ask: "What is a neurotransmitter?"
2. Show answer + top passages + highlights
3. Ask a harder question not in text
4. Show safe no-answer behavior
5. Close with architecture and strict grounding
