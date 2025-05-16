import os
import re
import glob
import json
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# ✅ Vue2 + TypeScript + class-component 기반 청크 추출기
def extract_chunks_from_vue(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # <template> 추출
    template_match = re.search(r"<template>(.*?)</template>", content, re.DOTALL)
    template = template_match.group(1).strip() if template_match else ""

    # <script> 추출
    script_match = re.search(r"<script[^>]*>(.*?)</script>", content, re.DOTALL)
    script = script_match.group(1).strip() if script_match else ""

    # 클래스명 추출
    class_match = re.search(r"export default class (\w+)", script)
    class_name = class_match.group(1) if class_match else "Unknown"

    # props 추출
    props = re.findall(r"@Prop\\(.*?\\)\\s+(\w+)!:.*?;", script)

    # emits 추출
    emits = re.findall(r'this\\.\\$emit\\(["\'](.*?)["\']', script)

    # methods 전체 블럭 추출
    method_blocks = re.findall(
        r"((?:@\w+\(.*?\)\s*)*(?:public\s+|private\s+|protected\s+)?(?:async\s+)?(?:get|set)?\s*(\w+)\s*\([^)]*\)\s*(?::\s*[^\{]+)?\{[\s\S]*?^\s*}\s*)",
        script,
        re.MULTILINE
    )

    chunks = []

    for full_block, method_name in method_blocks:
        chunk = {
            "file": file_path,
            "name": f"{class_name}.{method_name}",
            "code": full_block.strip()
        }
        chunks.append(chunk)

    return chunks


# ✅ 전체 디렉토리 대상 추출

def extract_vue_chunks_from_directory(root_dir):
    vue_files = glob.glob(os.path.join(root_dir, "**/*.vue"), recursive=True)
    all_chunks = []

    for file_path in vue_files:
        try:
            chunks = extract_chunks_from_vue(file_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"⚠️ {file_path} 파싱 실패: {e}")

    return all_chunks


# ✅ FastAPI + RAG 기반 검색 시스템
docs_path = "../repo/vue-ts-realworld-app"
embedding_model = HuggingFaceEmbeddings(model_name="../multilingual-e5-large-instruct")
vectorstore_path = "faiss_index"
json_output = "vue_code_chunks.json"

# ✅ 임베딩 처리 및 벡터 저장
if not os.path.exists(vectorstore_path):
    documents = extract_vue_chunks_from_directory(docs_path)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    texts = [doc['code'] for doc in documents]
    metadata = [{"file": doc['file'], "name": doc['name']} for doc in documents]
    db = FAISS.from_texts(texts, embedding_model, metadatas=metadata)
    db.save_local(vectorstore_path)
else:
    db = FAISS.load_local(vectorstore_path, embedding_model)

# ✅ LLM (예: codet5-base)
llm_model = AutoModelForSeq2SeqLM.from_pretrained("../../codet5-base")
tokenizer = AutoTokenizer.from_pretrained("../../codet5-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model.to(device)
llm_pipeline = pipeline("text2text-generation", model=llm_model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# ✅ QA 체인 구성
prompt = PromptTemplate.from_template("""
너는 Vue 2 + TypeScript + class-component로 작성된 프로젝트 코드를 요약하고 설명하는 AI야.

코드:
{context}

질문:
{question}

답변:
""")
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# ✅ FastAPI 실행
app = FastAPI()

class Query(BaseModel):
    question: str
    top_k: int = 3

@app.post("/search")
async def search_code(query: Query):
    try:
        result = rag_chain.run(query.question)
        return JSONResponse(content={
            "status": "ok",
            "question": query.question,
            "result": result
        }, ensure_ascii=False)
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

# ✅ 실행 메시지
if __name__ == "__main__":
    print("✅ FastAPI 서버를 uvicorn으로 실행하세요: uvicorn <this_file>:app --reload")
