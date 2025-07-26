import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class RAGPipeline:    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "microsoft/DialoGPT-medium",
        cache_dir: str = "./models",
        device: str = None
    ):

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            cache_folder=str(self.cache_dir)
        )
        
        # Initialize LLM
        print("Loading language model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            cache_dir=str(self.cache_dir)
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            cache_dir=str(self.cache_dir),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        # Initialize text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Initialize vector database
        self.vector_db = None
        self.documents = []
        self.document_embeddings = None
        
        print("RAG Pipeline initialized successfully!")
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):

        print(f"Adding {len(documents)} documents to the knowledge base...")
        
        # Store documents and metadata
        if metadata is None:
            metadata = [{"id": i} for i in range(len(documents))]
            
        for i, doc in enumerate(documents):
            self.documents.append({
                "text": doc,
                "metadata": metadata[i] if i < len(metadata) else {"id": len(self.documents)}
            })
        
        # Generate embeddings
        print("Generating embeddings...")
        all_texts = [doc["text"] for doc in self.documents]
        embeddings = self.embedding_model.encode(all_texts, show_progress_bar=True)
        
        # Create or update FAISS index
        if self.vector_db is None:
            dimension = embeddings.shape[1]
            self.vector_db = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.vector_db.add(embeddings.astype(np.float32))
        
        print(f"Knowledge base now contains {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:

        if self.vector_db is None or len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in vector database
        scores, indices = self.vector_db.search(query_embedding.astype(np.float32), top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                results.append({
                    "document": self.documents[idx],
                    "score": float(score),
                    "text": self.documents[idx]["text"]
                })
        
        return results
    
    def generate_response(
        self, 
        query: str, 
        context_docs: List[Dict], 
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        # Prepare context
        context_text = ""
        if context_docs:
            context_texts = [doc["text"][:500] for doc in context_docs]  # Limit context length
            context_text = "\n\n".join(context_texts)
        
        # Create prompt
        prompt = f"""Context information:
{context_text}

Question: {query}
Answer:"""
        
        try:
            # Generate response
            response = self.generator(
                prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )
            
            # Extract generated text
            generated_text = response[0]["generated_text"]
            
            # Extract only the answer part
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
                
            return answer
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(
        self, 
        question: str, 
        top_k: int = 3, 
        max_length: int = 200,
        include_sources: bool = True
    ) -> Dict[str, Any]:

        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        # Generate response
        answer = self.generate_response(question, retrieved_docs, max_length=max_length)
        
        # Prepare result
        result = {
            "question": question,
            "answer": answer,
        }
        
        if include_sources:
            result["sources"] = [
                {
                    "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                    "score": doc["score"],
                    "metadata": doc["document"]["metadata"]
                }
                for doc in retrieved_docs
            ]
        
        return result
    
    def save_knowledge_base(self, filepath: str):
        """Save the knowledge base to disk."""
        if self.vector_db is not None:
            # Save FAISS index
            faiss.write_index(self.vector_db, f"{filepath}.faiss")
            
            # Save documents
            with open(f"{filepath}_docs.json", "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
            
            print(f"Knowledge base saved to {filepath}")
    
    def load_knowledge_base(self, filepath: str):
        """Load the knowledge base from disk."""
        try:
            # Load FAISS index
            self.vector_db = faiss.read_index(f"{filepath}.faiss")
            
            # Load documents
            with open(f"{filepath}_docs.json", "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            
            print(f"Knowledge base loaded from {filepath}")
            print(f"Loaded {len(self.documents)} documents")
            
        except FileNotFoundError:
            print(f"Knowledge base files not found at {filepath}")
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")


def main():
    """Example usage of the RAG pipeline."""
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name="microsoft/DialoGPT-medium"  # Smaller model for demo
    )
    
    # Sample documents (you can replace with your own data)
    documents = [
        "Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development.",
        "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
        "Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
        "The transformer architecture is a neural network architecture that relies entirely on attention mechanisms to draw global dependencies between input and output."
    ]
    
    metadata = [
        {"source": "python_docs", "topic": "programming"},
        {"source": "ml_guide", "topic": "machine_learning"},
        {"source": "nlp_intro", "topic": "natural_language_processing"},
        {"source": "dl_basics", "topic": "deep_learning"},
        {"source": "transformer_paper", "topic": "deep_learning"}
    ]
    
    # Add documents to knowledge base
    rag.add_documents(documents, metadata)
    
    # Save knowledge base
    rag.save_knowledge_base("./knowledge_base")
    
    # Example queries
    queries = [
        "What is Python programming language?",
        "Explain machine learning",
        "What is NLP?",
        "Tell me about transformers in deep learning"
    ]
    
    print("\n" + "="*50)
    print("RAG PIPELINE DEMO")
    print("="*50)
    
    for query in queries:
        print(f"\nQuestion: {query}")
        print("-" * 30)
        
        result = rag.query(query, top_k=2, max_length=150)
        
        print(f"Answer: {result['answer']}")
        
        if result.get('sources'):
            print("\nSources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. Score: {source['score']:.3f} - {source['text']}")


if __name__ == "__main__":
    main()
