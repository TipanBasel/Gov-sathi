import faiss
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# =============================
# 1️⃣ Load embedding model
# =============================
embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# =============================
# 2️⃣ Load FAISS index
# =============================
index = faiss.read_index("govsathi_faiss.index")

# =============================
# 3️⃣ Load chunks
# =============================
with open("govsathi_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data["chunks"]
print(f"✅ Loaded {len(chunks)} chunks")

# =============================
# 4️⃣ Retriever (with score)
# =============================
def retrieve_chunks(question, k=5):
    q_vec = embed_model.encode([question])
    faiss.normalize_L2(q_vec)

    scores, indices = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "content": chunks[int(idx)]["content"]
        })

    return results

# =============================
# 5️⃣ Build Nepali Prompt (FIXED)
# =============================
def build_prompt(question, retrieved):
    context = "\n\n".join(
        [f"CHUNK {i+1}:\n{r['content']}" for i, r in enumerate(retrieved)]
    )

    return f"""
तपाईं नेपाल सरकारसम्बन्धी कानुनी जानकारी दिने AI सहायक हुनुहुन्छ।

नियम:
- तलका CHUNKS प्रश्नसँग सम्बन्धित छन् भने,
  तिनै CHUNKS बाट उत्तर संक्षेपमा लेख्नुहोस्।
- CHUNKS मा भएको कुरालाई **आफ्नै शब्दमा सारांश** बनाउनुहोस्।
- बाहिरी ज्ञान प्रयोग नगर्नुहोस्।
- CHUNKS बिल्कुलै असम्बन्धित भएमा मात्र लेख्नुहोस्:
  "मलाई यसबारे जानकारी छैन।"

====================
CHUNKS:
{context}
====================

प्रश्न:
{question}

निर्देशन:
- उत्तर नेपाली भाषामा दिनुहोस्
- कानुनी शब्दहरू नबदल्नुहोस्
- अनावश्यक सूचना नथप्नुहोस्

उत्तर:
""".strip()

# =============================
# 6️⃣ Call Ollama
# =============================
def call_ollama(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma2:2b",   # or llama3
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        },
        timeout=120
    )

    return res.json().get("response", "").strip()

# =============================
# 7️⃣ Full RAG Pipeline
# =============================
def answer_question(question):
    retrieved = retrieve_chunks(question)

    print("\n🔍 Retrieved Chunks:")
    for i, r in enumerate(retrieved, 1):
        print(f"\n===== CHUNK {i} | score={r['score']:.3f} =====")
        print(r["content"][:400])

    # 🔴 Similarity guard (VERY IMPORTANT)
    if retrieved[0]["score"] < 0.45:
        return "मलाई यसबारे जानकारी छैन।"

    prompt = build_prompt(question, retrieved)
    answer = call_ollama(prompt)

    # Final cleanup
    if not answer:
        return "मलाई यसबारे जानकारी छैन।"

    return answer

# =============================
# 8️⃣ Test
# =============================
if __name__ == "__main__":
    q = "बहु बाटो इजाजतपत्र भनेको के हो?"
    print("\n🧠 GovSathi Final Answer:\n")
    print(answer_question(q))
