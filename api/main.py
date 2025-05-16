from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import requests
import torch
import faiss
import pickle
import numpy as np
import os
import re
from typing import List, Dict, Any
import logging

app = FastAPI()

# 모델 및 벡터 인덱스 로드
model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_with_ollama(question: str, context: str) -> str:
    """Generate response using Ollama's Mistral model."""
    try:
        # Vue 2 + class component 스타일에 특화된 프롬프트
        prompt = f"""You are a Vue 2 + TypeScript + class-component expert. Please answer the following question based on the provided code context.
IMPORTANT: 
- Focus ONLY on Vue 2 + class-component syntax (using vue-property-decorator)
- DO NOT use Vue 3 Composition API or Options API
- Use the provided code examples as reference
- If the examples are not relevant, explain the general method using class-component syntax
- Format your response in markdown with proper code blocks and explanations
- DO NOT include any headers or section titles in your response
- DO NOT include any references or code examples in your response

Context:
{context}

Question: {question}

Answer in markdown format:"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # 더 정확한 답변을 위해 낮춤
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_ctx": 4096
                }
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated")
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return "Error generating response from Ollama"
    except Exception as e:
        logger.error(f"Error in generate_with_ollama: {str(e)}")
        return f"Error: {str(e)}"

def extract_code_chunks(code: str) -> List[str]:
    # <script> 태그 내부 코드만 추출
    script_match = re.search(r'<script[^>]*>(.*?)</script>', code, re.DOTALL)
    if script_match:
        code = script_match.group(1)

    # class ... extends Vue { ... } 전체를 추출
    class_pattern = r'class\s+\w+\s+extends\s+Vue\s*{([\s\S]*?)}'
    class_matches = re.finditer(class_pattern, code)
    chunks = []
    for match in class_matches:
        class_body = match.group(0)  # 전체 class ... { ... }
        # @Prop이 하나라도 있으면 청크로 포함
        if re.search(r'@Prop', class_body):
            chunks.append(class_body)
    return chunks

def is_valid_code(code: str) -> bool:
    # 1. 너무 짧거나 (30자로 완화)
    if len(code.strip()) < 30:
        return False
    # 2. 동일 단어 반복 (5회 이상 반복으로 완화)
    if re.search(r"(class\s*){5,}", code):
        return False
    # 3. 주석/공백이 90% 이상이면 (완화)
    lines = code.splitlines()
    if len(lines) == 0:
        return False
    comment_lines = sum(1 for l in lines if l.strip().startswith("//") or l.strip() == "")
    if comment_lines / len(lines) > 0.9:
        return False
    return True

def extract_code_from_file(file_path: str) -> List[Dict[str, str]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            chunks = extract_code_chunks(code)
            return [{
                "file": file_path,
                "name": os.path.splitext(os.path.basename(file_path))[0],
                "code": chunk
            } for chunk in chunks if is_valid_code(chunk)]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def build_code_index():
    # Vue + TS 파일 경로 리스트
    target_dir = "repo/vue-ts-realworld-app/src"
    vue_ts_files = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(('.vue', '.ts')):
                vue_ts_files.append(os.path.join(root, file))

    # 코드 청크 추출 및 필터링
    chunks = []
    for file_path in vue_ts_files:
        file_chunks = extract_code_from_file(file_path)
        chunks.extend(file_chunks)

    # 메타데이터 저장
    with open("data/output_faiss/metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    # FAISS 인덱스 생성
    if chunks:
        # 코드 임베딩 생성
        code_texts = [chunk["code"] for chunk in chunks]
        embeddings = model.encode(code_texts, normalize_embeddings=True)
        
        # FAISS 인덱스 생성 및 저장
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype="float32"))
        faiss.write_index(index, "data/output_faiss/faiss.index")
        
        print(f"✅ 인덱스 생성 완료: {len(chunks)}개의 코드 청크")
    else:
        print("❌ 유효한 코드 청크가 없습니다.")

# FAISS 인덱스 및 메타데이터 로드
FAISS_INDEX_PATH = "data/output_faiss/faiss.index"
METADATA_PATH = "data/output_faiss/metadata.pkl"

try:
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        print(f"✅ FAISS 인덱스 로드 완료: {len(metadata)}개의 코드 스니펫")
    else:
        print("⚠️ 인덱스 파일이 없습니다. 새로 생성합니다...")
        os.makedirs("data/output_faiss", exist_ok=True)
        build_code_index()
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
except Exception as e:
    print(f"❌ FAISS 인덱스 로드 실패: {str(e)}")
    index = None
    metadata = []

# 요청 스키마
class SearchRequest(BaseModel):
    question: str
    top_k: int = 2

def generate_answer(question: str, code_chunks: list) -> str:
    # 코드 예시를 더 명확하게 포맷팅
    context = "\n\n".join([
        f"### Example {i+1} from {chunk['file']}\n```ts\n{chunk['code']}\n```" 
        for i, chunk in enumerate(code_chunks[:2])
    ])
    
    return generate_with_ollama(question, context)

def rewrite_query(question: str) -> str:
    # 검색 쿼리 재작성 시 Vue 2 + class component 스타일 강조
    prompt = f"""Convert the following question into a specific search query for Vue 2 + TypeScript + class-component code.
Focus on finding code that uses vue-property-decorator decorators like @Prop, @Watch, @Component, etc.

Question: {question}

Search query:"""
    
    return generate_with_ollama(question, "")

@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "ok", "message": "Server is running"},
        status_code=200
    )

@app.post("/search")
async def search(req: SearchRequest):
    try:
        # 0 사용자 질문 정제
        refined_question = rewrite_query(req.question)
        print(f"Refined question: {refined_question}")

        # 1. 임베딩 생성 (e5 계열은 query: 접두어 필요)
        query_prompt = f"query: {refined_question}"
        query_embedding = model.encode([query_prompt], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype="float32")

        # 2. FAISS 검색
        distances, indices = index.search(query_embedding, req.top_k)

        # 3. 결과 구성 (중복 제거)
        seen_codes = set()
        references = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= len(metadata):
                continue
            entry = metadata[idx]
            code = entry["code"]
            if code not in seen_codes:
                seen_codes.add(code)
                references.append({
                    "file": entry["file"],
                    "name": entry["name"],
                    "score": float(dist),
                    "code": code
                })

        # 4. LLM 응답 생성
        answer = generate_with_ollama(req.question, references)

        # 5. 마크다운 포맷팅
        reference_code = "\n".join([
            f"### Example {i+1} from {ref['file']}\n```ts\n{ref['code']}\n```" 
            for i, ref in enumerate(references)
        ])
        
        markdown_result = f"""## 🤖 LLM 응답

{answer}

## 📚 참조 코드
{reference_code}"""

        return JSONResponse(
            content={
                "question": req.question,
                "status": "ok",
                "result": markdown_result,
                "references": references
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={
                "question": req.question,
                "status": "error",
                "message": str(e)
            },
            status_code=500
        )
