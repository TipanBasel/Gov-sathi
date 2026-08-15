import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load JSON
with open("govsathi_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 🔧 FIX: extract real chunks
if "chunks" in data:
    chunks = data["chunks"]
elif "documents" in data:
    chunks = []
    for doc in data["documents"]:
        chunks.extend(doc["chunks"])
else:
    raise ValueError("Unknown JSON structure")

print("📄 Total chunks:", len(chunks))

# Load model
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Extract text
texts = [c["content"] for c in chunks]

# Create embeddings
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

# Build FAISS
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Save
faiss.write_index(index, "govsathi_faiss.index")
np.save("govsathi_embeddings.npy", embeddings)

print("✅ FAISS rebuilt successfully")
print("Vectors:", index.ntotal)
