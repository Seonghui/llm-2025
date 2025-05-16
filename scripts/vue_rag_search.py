import os
import faiss
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
import torch
from dataclasses import dataclass
from langchain.schema import Document

@dataclass
class SearchResult:
    rag_type: str
    confidence: float
    metadata: Dict
    assessed_score: float

class VueCodeRetriever:
    def __init__(self, model_path: str = "../../multilingual-e5-large-instruct"):
        """
        Initialize the code retriever with E5 model
        Args:
            model_path: Path to the E5 model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for texts using E5 model
        """
        # Prepare the texts with the instruction prefix
        processed_texts = [f"query: {text}" for text in texts]
        
        # Tokenize and get embeddings
        with torch.no_grad():
            inputs = self.tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings.numpy()

    def add_documents(self, documents: List[Document]):
        """
        Add documents to the FAISS index
        Args:
            documents: List of Document objects containing code summaries
        """
        if not documents:
            return
            
        # Get texts and create embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self._get_embeddings(texts)
        
        # Initialize FAISS index if not exists
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(embeddings)
        self.documents.extend(documents)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve relevant code snippets for the query
        Args:
            query: Search query
            top_k: Number of results to return
        Returns:
            List of SearchResult objects
        """
        # Get query embedding
        query_embedding = self._get_embeddings([query])
        
        # Search in FAISS index
        if self.index is None or self.index.ntotal == 0:
            return []
            
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert distances to similarity scores (1 / (1 + distance))
        similarities = 1 / (1 + distances[0])
        
        # Prepare results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < len(self.documents):  # Safety check
                doc = self.documents[idx]
                result = SearchResult(
                    rag_type="faiss-vector",
                    confidence=float(similarity),
                    metadata={
                        "source": doc.metadata["uri"],
                        "title": doc.metadata["name"],
                        "excerpt": doc.page_content,
                        "code": doc.metadata["code"],
                        "type": doc.metadata["type"]
                    },
                    assessed_score=float(similarity)
                )
                results.append(result)
                
        return results

def main():
    # Example usage
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
    
    # Example search
    query = "사용자 인증 관련 컴포넌트 찾아줘"
    search_results = retriever.retrieve(query, top_k=3)
    
    # Print results
    print(f"\nSearch results for: {query}")
    for i, result in enumerate(search_results, 1):
        print(f"\n{i}. {result.metadata['title']}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Summary:\n{result.metadata['excerpt']}")
        print("-" * 80)

if __name__ == "__main__":
    main() 