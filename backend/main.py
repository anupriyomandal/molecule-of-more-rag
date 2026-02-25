import os
import re
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------
# CONFIG
# -----------------------------
TOP_K = 8
DISTANCE_THRESHOLD = 1.2
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# -----------------------------
# INIT
# -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Enable CORS (needed for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD VECTOR INDEX
# -----------------------------
try:
    index = faiss.read_index("vector.index")
    chunks = np.load("chunks.npy", allow_pickle=True)
    print("Vector index loaded successfully.")
except Exception as e:
    print("Error loading vector index:", e)
    index = None
    chunks = None

# -----------------------------
# UTIL: Clean Markdown
# -----------------------------
def clean_answer(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # remove bold
    text = text.replace("*", "")
    return text.strip()

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def distance_to_confidence(distance):
    # Lower L2 distance means better match.
    # Map distance to a bounded 0-100 score.
    raw = 1.0 - (distance / (DISTANCE_THRESHOLD * 2.0))
    return round(clamp(raw, 0.0, 1.0) * 100, 2)

def make_excerpt(text, max_chars=420):
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."

# -----------------------------
# REQUEST MODEL
# -----------------------------
class Query(BaseModel):
    question: str

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def root():
    return {"message": "RAG backend is running 🚀"}

@app.post("/ask")
def ask(query: Query):

    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if index is None or chunks is None:
        raise HTTPException(status_code=500, detail="Vector index is not loaded.")

    # 1️⃣ Embed Question
    embedding_response = client.embeddings.create(
        model=EMBED_MODEL,
        input=query.question
    )

    question_embedding = np.array(
        [embedding_response.data[0].embedding]
    ).astype("float32")

    # 2️⃣ Retrieve from FAISS
    distances, indices = index.search(question_embedding, TOP_K)
    retrieved_items = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < 0:
            continue
        chunk_text = str(chunks[idx])
        retrieved_items.append(
            {
                "chunk": chunk_text,
                "distance": float(distance),
                "confidence": distance_to_confidence(float(distance))
            }
        )

    if not retrieved_items:
        return {
            "answer": "I could not find this answer in the retrieved document context.",
            "mode": "RAG_NO_ANSWER",
            "confidence": 0.0,
            "sources_used": 0,
            "top_passages": []
        }

    retrieved_chunks = [item["chunk"] for item in retrieved_items]
    context = "\n\n".join(retrieved_chunks)

    avg_distance = float(np.mean([item["distance"] for item in retrieved_items]))
    overall_confidence = round(
        np.mean([item["confidence"] for item in retrieved_items[:5]]), 2
    )
    top_passages = [
        {
            "passage": make_excerpt(item["chunk"]),
            "distance": round(item["distance"], 4),
            "confidence": item["confidence"]
        }
        for item in retrieved_items[:5]
    ]

    # 3️⃣ Attempt RAG Answer
    rag_completion = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer strictly using the provided context. "
                    "If the answer is not present, respond with ONLY: NOT_FOUND"
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query.question}"
            }
        ]
    )

    rag_answer = rag_completion.choices[0].message.content.strip()

    # 4️⃣ Strict RAG: no fallback model call
    if "NOT_FOUND" in rag_answer or avg_distance > DISTANCE_THRESHOLD:
        return {
            "answer": "I could not find this answer in the retrieved document context.",
            "mode": "RAG_NO_ANSWER",
            "confidence": overall_confidence,
            "sources_used": len(top_passages),
            "top_passages": top_passages
        }

    # 5️⃣ Return RAG Answer
    return {
        "answer": clean_answer(rag_answer),
        "mode": "RAG",
        "confidence": overall_confidence,
        "sources_used": len(top_passages),
        "top_passages": top_passages
    }
