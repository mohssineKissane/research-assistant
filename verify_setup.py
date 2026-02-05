import sys
print(f"Python version: {sys.version}")

try:
    import langchain
    print(f"LangChain version: {langchain.__version__}")
except ImportError as e:
    print(f"Failed to import langchain: {e}")

try:
    import torch
    print(f"Torch version: {torch.__version__}")
except ImportError as e:
    print(f"Failed to import torch: {e}")

try:
    import chromadb
    print(f"ChromaDB version: {chromadb.__version__}")
except ImportError as e:
    print(f"Failed to import chromadb: {e}")

try:
    import sentence_transformers
    print(f"Sentence Transformers version: {sentence_transformers.__version__}")
except ImportError as e:
    print(f"Failed to import sentence_transformers: {e}")

print("Verification complete.")
