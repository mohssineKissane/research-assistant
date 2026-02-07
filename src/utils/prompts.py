"""
Prompt Templates - Instructions for the LLM
============================================

This module contains prompt templates that instruct the LLM how to
answer questions based on retrieved context.

What is a Prompt?
    The text sent to the LLM. It contains:
    - System instructions: How the LLM should behave
    - Context: Retrieved document chunks
    - Question: The user's question
    
Why Prompts Matter:
    The same LLM can behave very differently based on the prompt.
    Good prompts:
    - Tell the LLM to only use provided context (prevents hallucination)
    - Request citations (traceability)
    - Specify output format (structured responses)
    
Prompt Variables:
    LangChain prompts use {variable} placeholders that get filled at runtime.
    - {context}: The retrieved document chunks
    - {question}: The user's question
"""
from langchain.prompts import PromptTemplate, ChatPromptTemplate


class PromptTemplates:
    """
    Collection of prompt templates for different RAG use cases.
    
    Each template is a static method that returns a PromptTemplate.
    The templates instruct the LLM how to answer questions.
    
    LangChain's PromptTemplate:
    - Holds a template string with {variables}
    - format() fills in the variables
    - Used by chains to construct the final prompt
    """
    
    @staticmethod
    def get_qa_prompt():
        """
        Standard QA prompt with citation instructions.
        
        This prompt tells the LLM to:
        1. Only answer from the provided context
        2. Say "I don't have enough information" if answer isn't in context
        3. Include citations (document name and page number)
        4. Be concise but comprehensive
        
        Variables:
            {context}: The retrieved document chunks (filled by RetrievalQA)
            {question}: The user's question
            
        How it gets used:
            1. RetrievalQA retrieves relevant chunks
            2. Chunks are joined into the {context} string
            3. User's question fills {question}
            4. Complete prompt sent to LLM
            5. LLM generates text after "Answer:"
        """
        template = """You are a research assistant. Answer the question based on the provided context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the context provided
- If the answer isn't in the context, say "I don't have enough information"
- Include specific citations: mention the source document and page number
- Be concise but comprehensive

Answer:"""
        
        # Create PromptTemplate object
        # input_variables tells LangChain which placeholders to expect
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    @staticmethod
    def get_qa_with_sources_prompt():
        """
        Enhanced prompt that strongly emphasizes citations.
        
        Use this when you need:
        - Guaranteed citation format
        - Structured output (numbered sections)
        - Strict source attribution
        
        Output format:
            1. Direct answer (2-3 sentences)
            2. Supporting details with [Source: filename, Page: X] citations
        """
        template = """Answer the question using the context below. You MUST cite sources.

Context:
{context}

Question: {question}

Format your answer as:
1. Direct answer (2-3 sentences)
2. Supporting details with citations

Citation format: [Source: filename, Page: X]

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )