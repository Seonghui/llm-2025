from fastapi import FastAPI
from fastapi.responses import Response
import json

from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Optional


# ✅ 모델 & 인덱스 & 메타데이터 로드
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

model = SentenceTransformer("../../multilingual-e5-large-instruct")
index = faiss.read_index("../data/faiss.index")
tokenizer = AutoTokenizer.from_pretrained("../../gemma-ko-2b")
generator = AutoModelForCausalLM.from_pretrained("../../gemma-ko-2b")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 이부분 
generator.to(device)

with open("../data/chunks.pkl", "rb") as f:
    metadata = pickle.load(f)  # ✅ 우리 구조: { file, name, code }

# ✅ FastAPI 객체 생성
app = FastAPI()

# ✅ 요청 스키마
# 요청 모델
class SearchRequest(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.7
    max_length: int = 1024

# 응답 모델
class SearchResponse(BaseModel):
    question: str
    status: str
    result: str


# ✅ LLM 응답 생성
def create_prompt(question: str, relevant_chunks: List[dict]) -> str:
        # 시스템 프롬프트
        system_prompt = """당신은 Vue.js와 TypeScript에 대한 전문가입니다.
주어진 코드 예제들을 참고하여 사용자의 질문에 정확하고 자세하게 답변해주세요.
답변할 때는 다음 규칙을 따라주세요:
1. 코드 예제를 인용할 때는 구체적으로 어떤 파일의 어떤 부분인지 설명해주세요.
2. TypeScript와 Vue의 모범 사례를 따르는 설명을 제공해주세요.
3. 가능한 한 실제 코드 예제를 포함해서 설명해주세요.
4. 설명은 친절하고 이해하기 쉽게 작성해주세요."""

        # 관련 코드 컨텍스트 구성
        context = []
        for i, chunk in enumerate(relevant_chunks, 1):
            chunk_text = f"예제 {i}:\n"
            chunk_text += f"파일: {chunk['file_path']}\n"
            if chunk.get('semantic_type'):
                chunk_text += f"타입: {chunk['semantic_type']}\n"
            if chunk.get('context'):
                chunk_text += f"설명: {chunk['context']}\n"
            chunk_text += f"코드:\n{chunk['content']}\n"
            context.append(chunk_text)

        # 최종 프롬프트 구성
        prompt = f"{system_prompt}\n\n"
        prompt += "관련 코드 예제들:\n" + "\n---\n".join(context) + "\n\n"
        prompt += f"질문: {question}\n\n"
        prompt += "답변:"

        return prompt

async def search_and_generate(
        question: str,
        top_k: int = 5,
        temperature: float = 0.7,
        max_length: int = 1024
    ) -> str:
        try:
            # 1. 임베딩 생성
            query_embedding = model.encode(
                [question],
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            
            # 2. FAISS 검색
            distances, indices = index.search(
                query_embedding.astype('float32'),
                top_k
            )
            
            # 3. 관련 청크 수집
            relevant_chunks = []
            for idx, distance in zip(indices[0], distances[0]):
                if 0 <= idx < len(metadata):
                    chunk = metadata[idx]
                    relevant_chunks.append(chunk)
            
            # 4. 프롬프트 생성
            prompt = create_prompt(question, relevant_chunks)
            
            # 5. Gemma로 답변 생성
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)            
            with torch.no_grad():
                outputs = generator.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 6. 응답 추출 및 정리
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 프롬프트 이후의 텍스트만 추출
            answer = answer.split("답변:")[-1].strip()
            
            return answer
            
        except Exception as e:
            print(f"Error in search_and_generate: {e}")
            raise

# ✅ 검색 API
@app.post("/search")
async def search(request: SearchRequest):
    try:
        # 답변 생성
        result = await search_and_generate(
            request.question,
            request.top_k,
            request.temperature,
            request.max_length
        )

        return SearchResponse(
            question=request.question,
            status="ok",
            result=result
        )

    except Exception as e:
        print(f"Error in search endpoint: {e}")
        return SearchResponse(
            question=request.question,
            status="error",
            result=f"오류가 발생했습니다: {str(e)}"
        )