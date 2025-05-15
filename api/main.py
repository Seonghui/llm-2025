from fastapi import FastAPI
from fastapi.responses import Response
import json

from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ✅ 모델 & 인덱스 & 메타데이터 로드
# TODO: 모델  로컬, 원격 받게 수정 
# model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
model = SentenceTransformer("../../multilingual-e5-large-instruct")
index = faiss.read_index("../data/output_faiss/faiss.index")
tokenizer = AutoTokenizer.from_pretrained("../../codet5-base")
generator = AutoModelForSeq2SeqLM.from_pretrained("../../codet5-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)

with open("../data/output_faiss/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)  # ✅ 우리 구조: { file, name, code }

# ✅ FastAPI 객체 생성
app = FastAPI()

# ✅ 요청 스키마
class RAGRequest(BaseModel):
    question: str
    top_k: int = 2

# ✅ LLM 질문 생성 
def rewrite_query(question: str, embedding_model_name="intfloat/multilingual-e5-large-instruct", vector_db="FAISS") -> str:
    # 프롬프트: 벡터 모델과 DB 정보를 같이 제공
    prompt = f"""
다음 사용자의 질문을 벡터 검색에 적합하게 짧고 명확한 검색 쿼리로 바꿔줘.

사용자 질문: {question}

검색용 쿼리:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = generator.generate(
        inputs["input_ids"],
        max_new_tokens=64,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ LLM 응답 생성
def generate_answer_with_codet5(question: str, context_chunks: list[str]) -> str:
    # ✅ 시스템 역할 고정
    base_role = f"""너는 Vue 2 + TypeScript + class-component 기반 프로젝트의 AI 코드 비서야.
아래 코드를 참고해서 '{question}' 에 대해 정확하고 간결하게 답변해줘."""
    # ✅ 청크 수 제한
    limited_chunks = context_chunks[:2]  # 너무 많으면 truncate
    context = "\n\n".join([f"// {i+1}번 코드\n{chunk}" for i, chunk in enumerate(limited_chunks)])

    prompt = f"{base_role}\n\n{context}\n\n답변:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = generator.generate(
        inputs["input_ids"],
        max_new_tokens=256,
        num_beams=4,
        early_stopping=True
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean = ''.join(filter(lambda x: x.isprintable(), result))
    clean = clean.replace('\ufffd', '').strip()

    return clean


# ✅ 검색 API
@app.post("/search")
async def search(req: RAGRequest):
    try:
        # 0 사용자 질문 정제 
        refined_question = rewrite_query(req.question)
        print(refined_question)

        # 1. 임베딩 생성 (e5 계열은 query: 접두어 필요)
        query_prompt = f"query: {refined_question}"
        query_embedding = model.encode([query_prompt], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype="float32")

        # 2. FAISS 검색
        distances, indices = index.search(query_embedding, req.top_k)

        # 3. 결과 구성 (LLM 응답 생성)
        chunks = []
        references = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= len(metadata):
                continue
            entry = metadata[idx]
            chunks.append(entry["code"])
            references.append({
                "file": entry["file"],
                "name": entry["name"],
                "score": float(dist),
                "code": entry["code"]
            })
        answer = generate_answer_with_codet5(req.question, chunks)


        # ✅ 마크다운 포함한 응답 텍스트만 전달 (참고코드 없이)
        markdown_result = f"## 🤖 LLM 응답\n\n{answer.strip()}"

        return Response(
            content=json.dumps({
                "question": req.question,
                "status": "ok",
                "result": markdown_result
            }, ensure_ascii=False),
            media_type="application/json"
        )

        # 클라이언트에서 사용 시, document.getElementById("output").innerHTML = marked.parse(md);


        """ 
        마크다운 형식 응답 
        # 4. Markdown 형식의 응답 문자열 구성
        md_lines = [f"# 질문: {req.question}\n", "## 🤖 LLM 응답", answer, "\n## 🔍 참고 코드 (Top {})\n".format(len(references))]

        for i, ref in enumerate(references, 1):
            md_lines.append(f"### {i}. 🔹 {ref['name']} ({ref['file']})")
            md_lines.append(f"**Score:** {ref['score']:.4f}")
            md_lines.append("```ts")
            md_lines.append(ref["code"])
            md_lines.append("```\n")

        md_result = "\n".join(md_lines)

        # 5. 응답 반환 (Markdown 문자열을 results 필드로)
        return Response(content=md_result, media_type="text/markdown")

        """ 



    except Exception as e:
        # ❗에러 처리
        return {
            "question": req.question,
            "status": "error",
            "message": str(e)
        }
