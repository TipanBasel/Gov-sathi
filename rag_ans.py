import faiss
import json
import requests
import re
from sentence_transformers import SentenceTransformer

# Load embedding model (Nepali-friendly)

embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


#  Load FAISS index

index = faiss.read_index("govsathi_faiss.index")


#  Load chunks

with open("govsathi_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

if "chunks" in data:
    chunks = data["chunks"]
elif "documents" in data:
    chunks = []
    for doc in data["documents"]:
        chunks.extend(doc["chunks"])

print(f"✅ Loaded {len(chunks)} chunks")


#  Retriever

def retrieve_chunks(question, k=5):
    q_vec = embed_model.encode([question])
    faiss.normalize_L2(q_vec)

    _, indices = index.search(q_vec, k)
    return [chunks[int(i)] for i in indices[0]]


# Strong Nepali-only Prompt

def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(
        [f"CHUNK {i+1}:\n{ch['content']}" for i, ch in enumerate(retrieved_chunks)]
    )

    return f"""
तपाईं नेपाल सरकार सम्बन्धी जानकारी दिने AI सहायक हुनुहुन्छ।

⚠️ कडा नियम:
- उत्तर CHUNKS बाट मात्र लिनुहोस्
- बाहिरी ज्ञान, सुझाव, चेतावनी, Note, Disclaimer नलेख्नुहोस्
- अंग्रेजी शब्द प्रयोग नगर्नुहोस्
- यदि उत्तर छैन भने तलको वाक्य मात्र लेख्नुहोस्:
  "मलाई यसबारे जानकारी छैन।"

====================
CHUNKS:
{context}
====================

प्रश्न:
{question}

उत्तर:
""".strip()

# Call Ollama

def call_ollama(prompt):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.9
                }
            },
            timeout=120
        )
        return res.json().get("response", "").strip()
    except Exception as e:
        return ""


#  Post-clean (CRITICAL FIX)

def clean_answer(answer):
    if not answer:
        return ""

    lines = answer.splitlines()
    clean_lines = []

    for line in lines:
        #  Remove English
        if re.search(r"[A-Za-z]", line):
            continue

        #  Remove disclaimers / notes
        if any(x in line for x in ["Note", "consult", "authority", "best to"]):
            continue

        #  Remove fallback if mixed
        if "मलाई यसबारे जानकारी छैन" in line:
            continue

        clean_lines.append(line.strip())

    return "\n".join(l for l in clean_lines if l)


#  Full RAG Pipeline

def answer_question(question):
    retrieved = retrieve_chunks(question)

    print("\n🔍 Retrieved Chunks:")
    for i, ch in enumerate(retrieved, 1):
        print(f"\n========== CHUNK {i} ==========")
        print(ch["content"][:500])

    prompt = build_prompt(question, retrieved)
    raw_answer = call_ollama(prompt)
    final_answer = clean_answer(raw_answer)

    return final_answer if final_answer else "मलाई यसबारे जानकारी छैन।"


# Test

if __name__ == "__main__":
    q = "नयाँ सवारी चालक अनुमतिपत्र प्राप्त गर्न के के कागजात चाहिन्छ?"
    print("\n🧠 GovSathi Final Answer:\n")
    print(answer_question(q))
