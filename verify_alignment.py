import faiss
import json

# Load FAISS
index = faiss.read_index("govsathi_faiss.index")

# Load chunks
with open("govsathi_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("🔢 FAISS vectors:", index.ntotal)
print("📄 Chunk entries:", len(chunks))

# HARD CHECK
if index.ntotal != len(chunks):
    raise ValueError("❌ FAISS and chunks count DO NOT MATCH")
else:
    print("✅ FAISS and chunks are perfectly aligned")
