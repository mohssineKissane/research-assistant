"""
Retrieval QA Chain - The Heart of RAG Question-Answering
=========================================================

This module wraps LangChain's RetrievalQA chain to provide a clean interface
for question-answering with document retrieval.

How RetrievalQA Works:
    1. User asks: "What is AI?"
    2. Question → Embeddings model → Query vector [0.15, -0.32, ...]
    3. Query vector → ChromaDB → Find top-k similar chunks
    4. Chunks combined into context string
    5. Prompt template filled: context + question
    6. LLM generates answer based on context
    7. Return: answer + source documents

Chain Types in LangChain:
    - "stuff": Put ALL retrieved chunks in one prompt (what we use)
    - "map_reduce": Process chunks separately, then combine
    - "refine": Iteratively refine answer with each chunk
    - "map_rerank": Score each chunk's answer, pick best
    
We use "stuff" because it's simple and works well for small context windows.
"""
from langchain.chains import RetrievalQA
from src.utils.prompts import PromptTemplates


class RetrievalQAChain:
    """
    Wrapper around LangChain's RetrievalQA chain.
    
    This class simplifies the creation and usage of a retrieval-augmented
    question-answering system. It:
    
    1. Takes an LLM and a vectorstore
    2. Creates a retriever that searches the vectorstore
    3. Builds a RetrievalQA chain that combines retrieval + generation
    4. Provides a simple ask() interface
    
    The key insight: The retriever uses the SAME embeddings model
    that was used to index documents. This is how semantic search works -
    questions and documents are in the same vector space.
    """
    
    def __init__(self, llm, vectorstore, prompt_template=None):
        """
        Initialize the QA chain with components.
        
        Args:
            llm: LangChain LLM instance (e.g., ChatGroq)
                 This is what generates the answer text.
                 
            vectorstore: LangChain vectorstore (e.g., Chroma)
                         Contains embedded document chunks.
                         IMPORTANT: The vectorstore has the embeddings model
                         stored inside it - this is how questions get embedded.
                         
            prompt_template: Optional custom prompt (PromptTemplate)
                            Defines how context and question are presented to LLM.
                            Default: Our research assistant prompt with citation instructions.
        """
        self.llm = llm
        self.vectorstore = vectorstore
        
        # Use custom prompt or default research assistant prompt
        # The prompt tells the LLM HOW to answer (cite sources, stay grounded)
        self.prompt_template = prompt_template or PromptTemplates.get_qa_prompt()
        
        # The actual LangChain chain - created by create_chain()
        self.chain = None
    
    def create_chain(self, k=4):
        """
        Create the RetrievalQA chain.
        
        This method:
        1. Creates a Retriever from the vectorstore
        2. Builds a RetrievalQA chain that connects retriever + LLM + prompt
        
        Args:
            k: Number of chunks to retrieve for each question.
               - k=3: Fast, less context
               - k=5: Good balance
               - k=10: More context but higher token cost
               
        Returns:
            The LangChain RetrievalQA chain instance
        
        How the Retriever Works:
            retriever = vectorstore.as_retriever()
            
            When you call retriever.get_relevant_documents("What is AI?"):
            1. The retriever uses vectorstore._embedding_function
            2. This is the SAME HuggingFaceEmbeddings object from indexing
            3. Question is embedded: "What is AI?" → [0.15, -0.32, ...]
            4. ChromaDB finds k nearest vectors using cosine similarity
            5. Returns k Document objects with page_content and metadata
        """
        # Create retriever from vectorstore
        # The retriever wraps the vectorstore and provides get_relevant_documents()
        # Under the hood: vectorstore.as_retriever() returns a VectorStoreRetriever
        # that has access to the embeddings model through vectorstore._embedding_function
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",      # Use cosine similarity (default)
            search_kwargs={"k": k}         # Return top k results
        )
        
        # Create the RetrievalQA chain
        # This is the main LangChain component that orchestrates everything
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,                   # LLM for generation (ChatGroq)
            
            chain_type="stuff",             # Combine all chunks into one prompt
            # Other options:
            # "map_reduce" - process chunks separately, combine results
            # "refine" - iteratively improve answer with each chunk
            # "map_rerank" - score answers from each chunk, pick best
            
            retriever=retriever,            # Our retriever that finds relevant chunks
            
            return_source_documents=True,   # Include source docs in response
            # This is crucial for citations - we need to know which docs were used
            
            chain_type_kwargs={
                "prompt": self.prompt_template  # Custom prompt with instructions
            }
        )
        
        return self.chain
    
    def ask(self, question: str):
        """
        Ask a question and get an answer with sources.
        
        This is where the magic happens. When you call this method:
        
        1. self.chain({"query": question}) is called
        2. LangChain's RetrievalQA does:
           a. retriever.get_relevant_documents(question)
              - Embeds question using vectorstore's embeddings model
              - Searches ChromaDB for similar vectors
              - Returns top-k Document objects
           b. Combines documents into context string
           c. Fills prompt: context + question
           d. Calls LLM with filled prompt
           e. Returns {"result": answer, "source_documents": [docs]}
        
        Args:
            question: Natural language question
            
        Returns:
            dict with:
            - "answer": The LLM's generated answer (string)
            - "sources": List of LangChain Document objects used as context
                        Each Document has:
                        - page_content: The chunk text
                        - metadata: {filename, page, chunk_id, upload_date}
        """
        if self.chain is None:
            raise ValueError("Chain not created. Call create_chain() first")
        
        # Call the chain with the question
        # This triggers the entire RAG pipeline:
        # question → embed → search → context → prompt → LLM → answer
        result = self.chain({"query": question})
        
        # Return simplified format
        return {
            "answer": result["result"],           # The LLM's answer text
            "sources": result["source_documents"]  # List of Document objects
        }