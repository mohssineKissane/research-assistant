"""
Research Agent - Autonomous Decision-Making with ReAct Pattern
================================================================

This module implements an intelligent agent that can autonomously decide
which tools to use to answer research questions.

ReAct Pattern (Reasoning + Acting):
    1. THOUGHT: Agent analyzes the question
    2. ACTION: Agent decides which tool to use
    3. OBSERVATION: Agent sees the tool's output
    4. Repeat until answer is found

Example Flow:
    User: "What does the paper say about neural networks, and are there recent developments?"
    
    Thought: Need to search documents first
    Action: search_documents("neural networks")
    Observation: Found information about neural networks in paper...
    
    Thought: Now need recent developments from web
    Action: search_web("recent neural network developments 2024")
    Observation: Found recent articles...
    
    Thought: I have enough information to answer
    Final Answer: [Combines both sources]
"""
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.agents import Tool
from typing import List

class ResearchAgent:
    """
    Autonomous research agent that selects and uses tools to answer questions.
    
    The agent uses the ReAct (Reasoning + Acting) pattern to:
    - Reason about what information is needed
    - Act by calling appropriate tools
    - Observe the results
    - Iterate until the question is answered
    """
    
    def __init__(self, llm, vectorstore, tools_list=None):
        """
        Initialize the research agent.
        
        Args:
            llm: Language model for agent reasoning (decides which tools to use)
            vectorstore: ChromaDB instance for document search tool
            tools_list: Optional custom list of Tool objects. If None, creates default tools.
        """
        self.llm = llm
        self.vectorstore = vectorstore
        # Use provided tools or create standard set (document search, web search, summarization)
        self.tools = tools_list or self._create_default_tools()
        self.agent = None
    
    def _create_default_tools(self) -> List[Tool]:
        """
        Create the standard research tool set.
        
        Tools are the "actions" the agent can take. Each tool has:
        - name: Identifier for the tool
        - description: Tells the agent WHEN to use this tool (critical for tool selection)
        - _run(): The actual function that executes
        
        Returns:
            List of initialized tool instances
        """
        from src.tools.document_search import DocumentSearchTool
        from src.tools.web_search import WebSearchTool
        from src.tools.summarization import SummarizationTool
        
        tools = [
            # Search uploaded PDFs - agent uses this for document-specific questions
            DocumentSearchTool(vectorstore=self.vectorstore, llm=self.llm),
            # Search the web - agent uses this for current events or missing information
            WebSearchTool(max_results=5),
            # Summarize content - agent uses this when user asks for summaries
            SummarizationTool(llm=self.llm, vectorstore=self.vectorstore)
        ]
        
        return tools
    
    def create_agent(self, agent_type="zero-shot-react-description", verbose=True):
        """
        Create and initialize the agent executor.
        
        The agent executor is the "brain" that:
        1. Receives a question
        2. Decides which tool to use based on tool descriptions
        3. Executes the tool
        4. Analyzes the result
        5. Decides next action (use another tool or give final answer)
        
        Args:
            agent_type: Type of agent reasoning pattern
                - "zero-shot-react-description": Uses tool descriptions to decide (recommended)
                - "conversational-react-description": Same but with conversation memory
                - "react-docstore": Specialized for document Q&A
            verbose: If True, prints the agent's reasoning steps (useful for debugging)
        
        Returns:
            Initialized agent executor ready to answer questions
        """
        # Map string agent type to LangChain AgentType enum
        agent_type_map = {
            "zero-shot-react-description": AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            "conversational-react-description": AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            "react-docstore": AgentType.REACT_DOCSTORE
        }
        
        selected_agent_type = agent_type_map.get(
            agent_type, 
            AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
        
        # Initialize the agent with tools and configuration
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=selected_agent_type,
            verbose=verbose,  # Shows "Thought/Action/Observation" loop when True
            max_iterations=5,  # Prevent infinite loops (max 5 tool calls)
            early_stopping_method="generate",  # Stop when confident answer is found
            handle_parsing_errors=True  # Gracefully handle malformed LLM outputs
        )
        
        return self.agent
    
    def run(self, query: str) -> str:
        """
        Execute the agent on a query and return the final answer.
        
        The agent will autonomously:
        1. Analyze the query to understand what's needed
        2. Select appropriate tool(s) based on descriptions
        3. Execute tool(s) and observe results
        4. Synthesize a final answer from tool outputs
        
        Args:
            query: Natural language question or instruction
        
        Returns:
            String containing the final answer
            
        Example:
            >>> agent.run("What does the paper say about transformers?")
            "According to the paper, transformers are..."
        """
        if self.agent is None:
            raise ValueError("Agent not created. Call create_agent() first")
        
        try:
            # Run the agent - this triggers the ReAct loop
            result = self.agent.run(query)
            return result
        except Exception as e:
            return f"Agent error: {str(e)}"
    
    def invoke(self, query: str) -> dict:
        """
        Alternative to run() that returns detailed execution information.
        
        Unlike run() which returns just the answer string, invoke() returns:
        - input: The original query
        - output: The final answer
        - intermediate_steps: List of (action, observation) tuples showing tool usage
        
        Useful for debugging or understanding the agent's decision-making process.
        
        Args:
            query: Natural language question or instruction
            
        Returns:
            Dict with detailed execution information
        """
        if self.agent is None:
            raise ValueError("Agent not created")
        
        try:
            result = self.agent.invoke({"input": query})
            return result
        except Exception as e:
            return {"error": str(e)}