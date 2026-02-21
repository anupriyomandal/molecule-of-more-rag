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

    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    avg_distance = float(np.mean(distances))

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

    # 4️⃣ Fallback to GPT if needed
    if "NOT_FOUND" in rag_answer or avg_distance > DISTANCE_THRESHOLD:

        gpt_completion = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.5,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful neuroscience assistant."
                },
                {
                    "role": "user",
                    "content": query.question
                }
            ]
        )

        gpt_answer = gpt_completion.choices[0].message.content

        return {
            "answer": clean_answer(gpt_answer),
            "mode": "GPT"
        }

    # 5️⃣ Return RAG Answer
    return {
        "answer": clean_answer(rag_answer),
        "mode": "RAG"
    }