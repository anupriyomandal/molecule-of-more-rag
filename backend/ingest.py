import os
import faiss
import numpy as np
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# CONFIG
# ----------------------------
PDF_PATH = "../data/document.pdf"
CHUNK_SIZE = 1200        # characters per chunk
OVERLAP = 300            # overlap between chunks
BATCH_SIZE = 100         # embedding batch size
EMBED_MODEL = "text-embedding-3-small"

# ----------------------------
# INIT
# ----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Loading PDF...")

# ----------------------------
# LOAD PDF
# ----------------------------
reader = PdfReader(PDF_PATH)
text = ""

for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted + "\n"

print(f"Total characters extracted: {len(text)}")

# Basic cleanup
text = text.replace("\n\n", "\n")
text = text.strip()

# ----------------------------
# CHUNKING WITH OVERLAP
# ----------------------------
print("Creating chunks...")

chunks = []

for i in range(0, len(text), CHUNK_SIZE - OVERLAP):
    chunk = text[i:i + CHUNK_SIZE]
    if len(chunk.strip()) > 200:
        chunks.append(chunk)

print(f"Total chunks created: {len(chunks)}")

# ----------------------------
# CREATE EMBEDDINGS (BATCHED)
# ----------------------------
print("Generating embeddings...")

all_embeddings = []

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=batch
    )

    batch_embeddings = [item.embedding for item in response.data]
    all_embeddings.extend(batch_embeddings)

    print(f"Embedded {min(i+BATCH_SIZE, len(chunks))} / {len(chunks)} chunks")

# Convert to numpy
embeddings = np.array(all_embeddings).astype("float32")

# ----------------------------
# CREATE FAISS INDEX
# ----------------------------
print("Building FAISS index...")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ----------------------------
# SAVE FILES
# ----------------------------
faiss.write_index(index, "vector.index")
np.save("chunks.npy", np.array(chunks))

print("✅ Ingestion complete.")
print(f"Vector index saved with {len(chunks)} chunks.")