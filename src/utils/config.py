# Load environment variables
# Read YAML config
# Validate API keys exist
# Expose config as singleton or class

class Config:
    def __init__(self):
        load_dotenv()
        # Load yaml
        # Set LLM, embeddings, vectorstore configs
        self.llm = os.getenv("LLM")
        self.embeddings = os.getenv("EMBEDDINGS")
        self.vectorstore = os.getenv("VECTORSTORE")
    
    @property
    def groq_api_key(self):
         return os.getenv("GROQ_API_KEY")

    
    @property
    def chunk_size(self):
        # Return from yaml
        return os.getenv("CHUNK_SIZE")