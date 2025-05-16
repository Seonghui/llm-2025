import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ 임베딩 모델 로드 (multilingual-e5-large-instruct)
model = SentenceTransformer("../../multilingual-e5-large-instruct")
print("✅ 임베딩 모델 로드 완료")

# ✅ vue_chunks.json 로드
with open("../data/vue_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ✅ embeddingText 추출 및 전처리
texts = ["passage: " + chunk["embeddingText"] for chunk in chunks]

# ✅ 텍스트 임베딩
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

# ✅ FAISS 인덱스 생성
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(np.array(embeddings, dtype="float32"))

# ✅ 인덱스 저장
faiss.write_index(index, "../data/faiss.index")

# ✅ 메타데이터 저장
import pickle
metadata = [
    {
        "id": i,
        "file": chunk["filePath"],
        "name": chunk["className"],
        "code": chunk["embeddingText"]
    }
    for i, chunk in enumerate(chunks)
]

with open("../data/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ FAISS 인덱스 및 메타데이터 저장 완료")