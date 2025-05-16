import os
import traceback
from typing import List, Dict
from multiprocessing import Process, Pipe
from urllib import parse
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

def get_chat(profile_of_LLMs, selected_LLM):
    # Implement your chat model initialization here
    # This is a placeholder - you should implement based on your LLM setup
    pass

def summarize_vue_ts_chunks(chunks: List[str], file_path: str, profile_of_LLMs: List[Dict]) -> List[Document]:
    """
    Summarize Vue/TypeScript code chunks using parallel processing
    Args:
        chunks: List of code chunks to summarize
        file_path: Path to the source file
        profile_of_LLMs: List of LLM configurations
    Returns:
        List of Document objects containing summaries
    """
    selected_LLM = 0
    processes = []
    parent_connections = []
    
    for chunk in chunks:
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
        
        chat = get_chat(profile_of_LLMs, selected_LLM)
        bedrock_region = profile_of_LLMs[selected_LLM]['bedrock_region']
        
        # Determine file type for proper summarization
        file_type = 'vue' if file_path.endswith('.vue') else 'ts'
        
        process = Process(
            target=summarize_chunk_process,
            args=(child_conn, chat, chunk, file_path, file_type, bedrock_region)
        )
        processes.append(process)
        
        selected_LLM = (selected_LLM + 1) % len(profile_of_LLMs)
    
    # Start all processes
    for process in processes:
        process.start()
    
    # Collect results
    summaries = []
    for parent_conn in parent_connections:
        doc = parent_conn.recv()
        if doc:
            summaries.append(doc)
    
    # Wait for all processes to complete
    for process in processes:
        process.join()
    
    return summaries

def summarize_chunk_process(conn, chat, chunk, file_path, file_type, bedrock_region):
    """
    Process function to summarize a single code chunk
    """
    try:
        # Extract component/class/function name based on file type and content
        name = extract_code_identifier(chunk, file_type)
        if name:
            summary = summarize_code(chat, chunk, file_type)
            print(f"Summary ({bedrock_region}) for {name}: {summary}")
            
            # Remove name from summary if it starts with it
            if summary.startswith(name):
                summary = summary[summary.find('\n')+1:] if '\n' in summary else summary[len(name):].lstrip()
            
            doc = Document(
                page_content=summary,
                metadata={
                    'name': name,
                    'uri': file_path,
                    'code': chunk,
                    'type': file_type
                }
            )
            conn.send(doc)
        else:
            conn.send(None)
            
    except Exception:
        err_msg = traceback.format_exc()
        print('Error message: ', err_msg)
        conn.send(None)
    
    finally:
        conn.close()

def extract_code_identifier(chunk: str, file_type: str) -> str:
    """
    Extract the identifier (component/class/function name) from a code chunk
    """
    try:
        if file_type == 'vue':
            # Check for Vue class component patterns
            patterns = [
                ('@Component', '\n@Component', '\nexport default class'),
                ('class', '\nexport default class', '{'),
                ('script', '\n<script', '>'),
                ('template', '\n<template', '>'),
                ('style', '\n<style', '>')
            ]
        else:  # TypeScript
            patterns = [
                ('class', '\nclass ', '{'),
                ('interface', '\ninterface ', '{'),
                ('type', '\ntype ', '='),
                ('function', '\nfunction ', '('),
                ('const', '\nconst ', '='),
                ('let', '\nlet ', '='),
                ('export', '\nexport ', ' ')
            ]
        
        for pattern_name, start_pattern, end_pattern in patterns:
            if start_pattern in chunk:
                start_idx = chunk.find(start_pattern) + len(start_pattern)
                end_idx = chunk.find(end_pattern, start_idx)
                if end_idx != -1:
                    return chunk[start_idx:end_idx].strip()
        
        return None
        
    except Exception:
        return None

def summarize_code(chat, code: str, file_type: str) -> str:
    """
    Generate a summary of the code using the specified chat model
    """
    if file_type == 'vue':
        system = (
            "다음의 <article> tag에는 Vue.js 컴포넌트 코드가 있습니다. "
            "코드의 전반적인 목적에 대해 설명하고, 컴포넌트의 기능, 데코레이터, "
            "메서드, props, computed 속성 등의 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    else:  # TypeScript
        system = (
            "다음의 <article> tag에는 TypeScript 코드가 있습니다. "
            "코드의 전반적인 목적에 대해 설명하고, 타입 정의, 인터페이스, "
            "클래스, 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    
    chain = prompt | chat
    try:
        result = chain.invoke({"code": code})
        return result.content
    except Exception:
        err_msg = traceback.format_exc()
        print('Error message: ', err_msg)
        raise Exception("Unable to request summary from LLM")

def main():
    # Example usage
    from vue_splitter import load_vue_ts_files
    
    # Configure your LLM profiles
    profile_of_LLMs = [
        {
            'name': 'claude-v2',
            'bedrock_region': 'us-east-1',
            # Add other necessary configuration
        }
        # Add more LLM profiles as needed
    ]
    
    # Load and process files
    repo_path = "./repo/vue-ts-realworld-app"
    results = load_vue_ts_files(repo_path)
    
    # Process each file's chunks
    for result in results:
        print(f"\nProcessing file: {result['file_path']}")
        summaries = summarize_vue_ts_chunks(
            result['chunks'],
            result['file_path'],
            profile_of_LLMs
        )
        
        # Print summaries
        for summary in summaries:
            print(f"\nComponent/Function: {summary.metadata['name']}")
            print(f"Summary:\n{summary.page_content}")
            print("-" * 80)

if __name__ == "__main__":
    main() 