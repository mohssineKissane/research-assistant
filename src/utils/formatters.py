"""
Response Formatters - Making Answers User-Friendly
===================================================

This module formats RAG responses for display.
It extracts citations from source documents and creates clean output.

What We Receive from RAG:
    - answer: Raw text from the LLM
    - sources: List of LangChain Document objects that were used as context
    
What We Output:
    - Formatted answer
    - Deduplicated citations with source info
    - Pretty display for terminal/UI

LangChain Document objects:
    When RetrievalQA returns source_documents, each Document has:
    - page_content: The chunk text that was sent to the LLM
    - metadata: {filename, page, chunk_id, upload_date, source}
    
We extract this metadata to show the user WHERE answers came from.
"""
from typing import List
from langchain.schema import Document


class ResponseFormatter:
    """
    Utility class for formatting QA responses.
    
    Takes raw output from the QA chain and creates:
    1. Clean answer text
    2. Extracted citations with file/page info
    3. Display-ready formatted strings
    
    All methods are static - no instance needed.
    """
    
    @staticmethod
    def format_answer_with_sources(answer: str, sources: List[Document]) -> dict:
        """
        Format the answer and extract deduplicated citations.
        
        Args:
            answer: The LLM's generated answer text
            
            sources: List of Document objects used as context
                    Each Document has:
                    - page_content: The text chunk
                    - metadata: {filename, page, chunk_id, upload_date}
        
        Returns:
            dict with:
            - 'answer': The answer text (unchanged)
            - 'citations': List of citation dicts, each containing:
                          - filename: Source file name
                          - page: Page number in original PDF
                          - content_preview: First 200 chars of the chunk
                          - chunk_id: Which chunk this is
            - 'num_sources': Count of unique sources
            
        Why Deduplicate?
            Sometimes the same page appears in multiple chunks.
            We track seen sources to avoid showing duplicate citations.
            
        Example output:
            {
                'answer': 'AI is a branch of computer science...',
                'citations': [
                    {'filename': 'intro.pdf', 'page': 1, 'content_preview': 'What is AI...', 'chunk_id': 0},
                    {'filename': 'intro.pdf', 'page': 3, 'content_preview': 'Machine learning...', 'chunk_id': 5}
                ],
                'num_sources': 2
            }
        """
        citations = []
        seen_sources = set()  # Track unique sources to avoid duplicates
        
        for doc in sources:
            # Create unique identifier from filename + page
            source_id = f"{doc.metadata.get('filename', 'Unknown')}:page{doc.metadata.get('page', '?')}"
            
            # Only add if we haven't seen this exact source before
            if source_id not in seen_sources:
                citations.append({
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'page': doc.metadata.get('page', '?'),
                    'content_preview': doc.page_content[:200] + "...",  # First 200 chars
                    'chunk_id': doc.metadata.get('chunk_id')
                })
                seen_sources.add(source_id)
        
        return {
            'answer': answer,
            'citations': citations,
            'num_sources': len(citations)
        }
    
    @staticmethod
    def format_for_display(response: dict) -> str:
        """
        Create a nicely formatted string for terminal/UI display.
        
        Args:
            response: Output from format_answer_with_sources()
            
        Returns:
            Multi-line string with:
            - Section divider
            - Answer text
            - Section divider
            - Numbered list of sources with previews
            
        Example output:
            ============================================================
            ANSWER:
            AI is a branch of computer science that aims to create
            intelligent machines...
            
            ============================================================
            SOURCES (2):
            
            [1] intro.pdf (Page 1)
                Preview: What is AI? Artificial intelligence is...
            
            [2] intro.pdf (Page 3)
                Preview: Machine learning is a subset of AI that...
        """
        output = f"\n{'='*60}\n"
        output += f"ANSWER:\n{response['answer']}\n"
        output += f"\n{'='*60}\n"
        output += f"SOURCES ({response['num_sources']}):\n"
        
        for i, citation in enumerate(response['citations'], 1):
            output += f"\n[{i}] {citation['filename']} (Page {citation['page']})\n"
            output += f"    Preview: {citation['content_preview']}\n"
        
        return output