import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    print("Importing RAG Service...")
    from rag_service import RAGService
    print("Initializing (this downloads the model if needed)...")
    rag = RAGService()
    print("Testing query...")
    print(rag.query_rag("What tea do you have?"))
    print("Warmup successful!")
except Exception as e:
    print(f"Error during warmup: {e}")
    sys.exit(1)
