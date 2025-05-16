from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Optional
import os
from llama_cpp import Llama
import logging

# 로깅 설정
logging.basicConfig(level=logging.WARNING)

# 기본 설정
EMBEDDING_MODEL = "../../multilingual-e5-large-instruct"
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
LLM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/llama-2-ko-7b.gguf")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

class CodeAssistant:
    def __init__(self):
        # 임베딩 모델 초기화
        print("임베딩 모델 로딩 중...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        
        # LLM 모델 초기화
        print("Llama 모델 로딩 중...")
        self.llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=2048,
            n_threads=6,
            n_gpu_layers=0,
            verbose=False,
            use_mlock=True,
            use_mmap=True,
        )
        
        # FAISS 인덱스 & 청크 로드
        print("FAISS 인덱스 로딩 중...")
        faiss_path = os.path.join(DATA_DIR, "faiss.index")
        print(f"FAISS 인덱스 경로: {faiss_path}")
        self.index = faiss.read_index(faiss_path)
        
        print("청크 데이터 로딩 중...")
        chunks_path = os.path.join(DATA_DIR, "chunks.pkl")
        print(f"청크 데이터 경로: {chunks_path}")
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
            
        print("초기화 완료!")

    def create_prompt(self, question: str, relevant_chunks: List[dict]) -> str:
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
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.7,
        max_length: int = 1024
    ) -> str:
        try:
            # 1. 임베딩 생성
            query_embedding = self.embedding_model.encode(
                [question],
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            
            # 2. FAISS 검색
            distances, indices = self.index.search(
                query_embedding.astype('float32'),
                top_k
            )
            
            # 3. 관련 청크 수집
            relevant_chunks = []
            for idx, distance in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    relevant_chunks.append(chunk)
            
            # 4. 프롬프트 생성
            prompt = self.create_prompt(question, relevant_chunks)
            
            # 5. Llama로 답변 생성
            response = self.llm(
                prompt,
                max_tokens=max_length,
                temperature=temperature,
                top_p=0.95,
                repeat_penalty=1.2,
                echo=False
            )
            
            # 6. 응답 추출 및 정리
            answer = response["choices"][0]["text"].strip()
            answer = answer.split("답변:")[-1].strip()
            
            return answer
            
        except Exception as e:
            print(f"Error in search_and_generate: {e}")
            raise

# FastAPI 앱 초기화
app = FastAPI()

# 전역 assistant 인스턴스 생성
assistant = CodeAssistant()

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

@app.post("/search")
async def search(request: SearchRequest):
    try:
        # 답변 생성
        result = await assistant.search_and_generate(
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
