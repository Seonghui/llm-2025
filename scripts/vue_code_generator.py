import traceback
from typing import List, Dict
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from vue_rag_search import SearchResult

def priority_search(query: str, relevant_codes: List[SearchResult], top_k: int = 3) -> List[SearchResult]:
    """
    우선순위에 따라 관련 코드 선택
    """
    # 신뢰도 점수로 정렬
    sorted_codes = sorted(relevant_codes, key=lambda x: x.confidence, reverse=True)
    return sorted_codes[:top_k]

def create_context_from_codes(relevant_codes: List[SearchResult]) -> str:
    """
    관련 코드들을 하나의 컨텍스트로 결합
    """
    context = ""
    for doc in relevant_codes:
        if doc.metadata['code']:
            # 코드 블록에 파일 정보와 구분자 추가
            context += f"// File: {doc.metadata['source']}\n"
            context += f"// Component: {doc.metadata['title']}\n"
            context += doc.metadata['code']
            context += "\n\n" + "=" * 80 + "\n\n"
    
    return context

def generate_vue_code(chat, query: str, context: str, file_type: str = 'vue') -> str:
    """
    주어진 컨텍스트를 기반으로 새로운 Vue/TypeScript 코드 생성
    """
    if file_type == 'vue':
        system = """다음의 <context> tag안에는 질문과 관련된 Vue.js 컴포넌트 코드가 있습니다. 
        주어진 예제를 참조하여 질문과 관련된 Vue.js 컴포넌트 코드를 생성합니다. 
        Vue 2, TypeScript, class-component 스타일로 작성해주세요.
        Assistant의 이름은 서연입니다.
        
        <context>
        {context}
        </context>"""
    else:  # TypeScript
        system = """다음의 <context> tag안에는 질문과 관련된 TypeScript 코드가 있습니다. 
        주어진 예제를 참조하여 질문과 관련된 TypeScript 코드를 생성합니다.
        Vue 2, class-component 스타일에 맞는 TypeScript 코드를 작성해주세요.
        Assistant의 이름은 서연입니다.
        
        <context>
        {context}
        </context>"""
    
    human = "<question>{text}</question>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('Prompt template created')
    
    chain = prompt | chat
    try:
        print(f"Generating code for query: {query}")
        result = chain.invoke({
            "context": context,
            "text": query
        })
        
        generated_code = result.content
        print("Code generation completed")
        
    except Exception:
        err_msg = traceback.format_exc()
        print('Error message: ', err_msg)
        raise Exception("Unable to generate code using LLM")
    
    return generated_code

def main():
    # Example usage
    from vue_rag_search import VueCodeRetriever
    from vue_summarizer import load_vue_ts_files, summarize_vue_ts_chunks
    
    # Initialize retriever
    retriever = VueCodeRetriever()
    
    # Load and process files
    repo_path = "./repo/vue-ts-realworld-app"
    results = load_vue_ts_files(repo_path)
    
    # Process files and add to retriever
    for result in results:
        print(f"\nProcessing file: {result['file_path']}")
        summaries = summarize_vue_ts_chunks(
            result['chunks'],
            result['file_path'],
            []  # LLM profiles not needed for this example
        )
        retriever.add_documents(summaries)
    
    # Example search and code generation
    query = "사용자 프로필 수정 폼 컴포넌트 만들어줘"
    
    # 1. Retrieve relevant codes
    search_results = retriever.retrieve(query, top_k=5)
    
    # 2. Priority filtering
    selected_codes = priority_search(query, search_results, top_k=3)
    
    # 3. Create context
    context = create_context_from_codes(selected_codes)
    
    # 4. Generate new code
    # Note: You need to implement the chat model initialization
    chat = None  # Replace with your chat model
    if chat:
        generated_code = generate_vue_code(chat, query, context, 'vue')
        
        # Print results
        print(f"\nGenerated code for: {query}")
        print("=" * 80)
        print(generated_code)
        print("=" * 80)
    else:
        print("Chat model not initialized. Skipping code generation.")

if __name__ == "__main__":
    main() 