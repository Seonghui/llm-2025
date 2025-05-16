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
from langchain_ollama.llms import OllamaLLM

EMBEDDING_MODEL = "../../multilingual-e5-large-instruct"
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
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
        self.llm = OllamaLLM(model="gemma3:1b")
        
        # FAISS 인덱스 & 청크 로드
        print("FAISS 인덱스 로딩 중...")
        faiss_path = os.path.join(DATA_DIR, "faiss.index")
        print(f"FAISS 인덱스 경로: {faiss_path}")
        self.index = faiss.read_index(faiss_path)
        
        print("청크 데이터 로딩 중...")
        chunks_path = os.path.join(DATA_DIR, "metadata.pkl")
        print(f"청크 데이터 경로: {chunks_path}")
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
            
        print("초기화 완료!")

    def classify_question_type(self, question: str) -> str:
        # 질문 유형 분류를 위한 키워드
        summary_keywords = ["요약", "간단히", "짧게", "핵심만"]
        optimization_keywords = ["최적화", "개선", "리팩토링", "더 나은", "더 좋은"]
        generation_keywords = ["만들어", "생성", "새로", "추가"]

        question_lower = question.lower()
        
        for keyword in summary_keywords:
            if keyword in question:
                return "summary"
                
        for keyword in optimization_keywords:
            if keyword in question:
                return "optimization"
                
        for keyword in generation_keywords:
            if keyword in question:
                return "generation"
                
        return "general"  # 기본 유형

    def get_system_prompt(self, question_type: str) -> str:
        base_prompt = """당신은 Vue.js와 TypeScript에 대한 전문가입니다.
답변할 때는 다음 규칙을 따라주세요:
1. 코드 예제를 인용할 때는 구체적으로 어떤 파일의 어떤 부분인지 설명해주세요.
2. TypeScript와 Vue의 모범 사례를 따르는 설명을 제공해주세요.
3. 가능한 한 실제 코드 예제를 포함해서 설명해주세요.
4. 설명은 친절하고 이해하기 쉽게 작성해주세요."""

        if question_type == "summary":
            return base_prompt + """
주어진 코드를 다음과 같이 요약해주세요:
1. 코드의 핵심 기능과 목적
2. 주요 컴포넌트와 메서드의 역할
3. 중요한 로직이나 패턴
4. 전체 구조를 간단히 설명"""

        elif question_type == "optimization":
            return base_prompt + """
주어진 코드를 다음 관점에서 개선해주세요:
1. 성능 최적화 가능성
2. 코드 가독성과 유지보수성
3. TypeScript/Vue.js 모범 사례 적용
4. 잠재적인 버그나 문제점
5. 구체적인 개선 방안과 예시 코드"""

        elif question_type == "generation":
            return base_prompt + """
요청한 컴포넌트/기능을 다음 사항을 고려하여 생성해주세요:
1. TypeScript와 Vue.js의 최신 문법과 기능 활용
2. 재사용성과 확장성을 고려한 설계
3. 적절한 타입 정의와 인터페이스 사용
4. 에러 처리와 예외 상황 고려
5. 구체적인 구현 코드와 사용 예시"""

        else:
            return base_prompt

    def create_prompt(self, question: str, relevant_chunks: List[dict]) -> str:
        # 질문 유형 분류
        question_type = self.classify_question_type(question)
        
        # 질문 유형에 따른 시스템 프롬프트 선택
        system_prompt = self.get_system_prompt(question_type)

        # 관련 코드 컨텍스트 구성
        context = []
        for i, chunk in enumerate(relevant_chunks, 1):
            chunk_text = f"예제 {i}:\n"
            chunk_text += f"파일: {chunk['file']}\n"
            if chunk.get('name'):
                chunk_text += f"타입: {chunk['name']}\n"
            chunk_text += f"코드:\n{chunk['code']}\n"
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
            response = self.llm.invoke(prompt)
            print(response)
            
            # 6. 응답 추출 및 정리
            answer = response.strip()
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