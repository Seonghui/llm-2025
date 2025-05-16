import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_vue_ts_files(repo_path: str) -> List[dict]:
    """
    Load and split Vue TypeScript files from the repository
    Args:
        repo_path: Path to the repository
    Returns:
        List of dictionaries containing file info and chunks
    """
    results = []
    
    # Walk through the repository
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.vue', '.ts')):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        contents = f.read()
                    
                    # Different splitting strategies based on file type
                    if file.endswith('.vue'):
                        chunks = split_vue_file(contents)
                    else:  # .ts files
                        chunks = split_ts_file(contents)
                    
                    if chunks:
                        results.append({
                            'file_path': relative_path,
                            'chunks': chunks
                        })
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    
    return results

def split_vue_file(contents: str) -> List[str]:
    """
    Split Vue component file into logical chunks
    """
    # Vue file specific splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            # Class component specific separators
            "\n@Component",
            "\nexport default class",
            # Vue component sections
            "\n<template>",
            "\n<script>",
            "\n<style",
            # Method separators
            "\n  @Watch",
            "\n  @Prop",
            "\n  @Emit",
            "\n  public ",
            "\n  private ",
            "\n  protected ",
            # General code separators
            "\n\n",
            "\n",
            " ",
            ""
        ],
        length_function=len,
    )
    
    return splitter.split_text(contents)

def split_ts_file(contents: str) -> List[str]:
    """
    Split TypeScript file into logical chunks
    """
    # TypeScript specific splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            # TypeScript specific separators
            "\ninterface ",
            "\ntype ",
            "\nclass ",
            "\nfunction ",
            "\nexport ",
            "\nimport ",
            # General code separators
            "\n\n",
            "\n",
            " ",
            ""
        ],
        length_function=len,
    )
    
    return splitter.split_text(contents)

def main():
    # Replace this with your repository path
    repo_path = "./repo/vue-ts-realworld-app"
    
    # Process all files
    results = load_vue_ts_files(repo_path)
    
    # Print results
    for result in results:
        print(f"\nFile: {result['file_path']}")
        print(f"Number of chunks: {len(result['chunks'])}")
        print("Sample chunks:")
        for i, chunk in enumerate(result['chunks'][:2]):  # Print first 2 chunks as sample
            print(f"\nChunk {i+1}:")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

if __name__ == "__main__":
    main() 