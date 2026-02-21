# LangChain Agents: A Complete Developer's Guide

## Table of Contents
1. [What is an Agent?](#what-is-an-agent)
2. [The ReAct Pattern](#the-react-pattern)
3. [Agent Architecture](#agent-architecture)
4. [Tools: The Agent's Capabilities](#tools-the-agents-capabilities)
5. [Agent Types in LangChain](#agent-types-in-langchain)
6. [Building an Agent: Our Implementation](#building-an-agent-our-implementation)
7. [Agent vs Traditional Chains](#agent-vs-traditional-chains)
8. [Best Practices](#best-practices)
9. [Common Pitfalls](#common-pitfalls)
10. [Advanced Topics](#advanced-topics)

---

## What is an Agent?

### Definition

An **agent** is an AI system that can:
1. **Reason** about what to do
2. **Decide** which tools to use
3. **Execute** actions autonomously
4. **Iterate** until it has an answer

Think of it as giving your LLM "hands" to interact with the world.

### Agent vs LLM

```
┌─────────────────────────────────────────────────────────┐
│                    Traditional LLM                       │
│                                                          │
│  User Question → LLM → Answer                           │
│                                                          │
│  Limitation: Can only use knowledge from training       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    LLM with Agent                        │
│                                                          │
│  User Question → Agent → [Thinks] → [Uses Tool] →       │
│                  ↑         ↓                             │
│                  └─────[Observes Result]                 │
│                          ↓                               │
│                    Final Answer                          │
│                                                          │
│  Capability: Can search docs, web, run code, etc.       │
└─────────────────────────────────────────────────────────┘
```

### Key Concept: Autonomy

The agent **decides** which tools to use. You don't tell it "search the web" - it figures that out based on the question.

---

## The ReAct Pattern

### What is ReAct?

**ReAct** = **Rea**soning + **Act**ing

It's a prompting technique where the LLM alternates between:
- **Thought**: Reasoning about what to do next
- **Action**: Executing a tool
- **Observation**: Seeing the result

### ReAct Loop Example

```
User: "What does the paper say about AI, and what's new in 2024?"

┌─────────────────────────────────────────────────────────┐
│ Thought: I need to find info about AI in the paper     │
│ Action: search_documents("AI")                          │
│ Observation: [Found: "AI enables machines to learn..."]│
├─────────────────────────────────────────────────────────┤
│ Thought: Now I need current 2024 developments          │
│ Action: search_web("AI developments 2024")              │
│ Observation: [Found: "GPT-5 released, new models..."]  │
├─────────────────────────────────────────────────────────┤
│ Thought: I have enough information to answer            │
│ Final Answer: According to the paper, AI enables...    │
│               In 2024, major developments include...    │
└─────────────────────────────────────────────────────────┘
```

### Why ReAct Works

1. **Transparency**: You can see the agent's reasoning
2. **Debuggability**: Easy to spot where it went wrong
3. **Flexibility**: Can handle complex multi-step tasks
4. **Reliability**: Less prone to hallucination (uses real tools)

---

## Agent Architecture

### Components

```
┌──────────────────────────────────────────────────────────┐
│                    AGENT EXECUTOR                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │                      LLM                            │  │
│  │  (The "brain" - decides what to do)                │  │
│  └────────────────────────────────────────────────────┘  │
│                           ↓                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │                  AGENT PROMPT                       │  │
│  │  - Instructions on how to think                    │  │
│  │  - Tool descriptions                               │  │
│  │  - Examples of reasoning                           │  │
│  └────────────────────────────────────────────────────┘  │
│                           ↓                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │                    TOOLS                            │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │  │
│  │  │Document  │  │   Web    │  │Summarize │         │  │
│  │  │ Search   │  │  Search  │  │  Tool    │         │  │
│  │  └──────────┘  └──────────┘  └──────────┘         │  │
│  └────────────────────────────────────────────────────┘  │
│                           ↓                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │                EXECUTION LOOP                       │  │
│  │  1. Generate Thought                               │  │
│  │  2. Choose Action (tool)                           │  │
│  │  3. Execute Tool                                   │  │
│  │  4. Observe Result                                 │  │
│  │  5. Repeat or Answer                               │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Flow Diagram

```
Start
  ↓
User asks question
  ↓
Agent receives question + tool descriptions
  ↓
┌─────────────────────────┐
│ LLM generates "Thought" │
└─────────────────────────┘
  ↓
  ├─→ "I have enough info" → Generate Final Answer → END
  │
  └─→ "I need to use a tool"
       ↓
     ┌──────────────────────┐
     │ LLM chooses Action   │
     │ (which tool to use)  │
     └──────────────────────┘
       ↓
     ┌──────────────────────┐
     │ Execute Tool         │
     └──────────────────────┘
       ↓
     ┌──────────────────────┐
     │ Observe Result       │
     └──────────────────────┘
       ↓
     Loop back to "Thought"
```

---

## Tools: The Agent's Capabilities

### What is a Tool?

A **tool** is a function the agent can call. Each tool has:
1. **Name**: Identifier (e.g., `search_documents`)
2. **Description**: Tells agent WHEN to use it
3. **Input Schema**: What parameters it needs
4. **Implementation**: The actual code that runs

### Tool Anatomy

```python
from langchain.tools import BaseTool
from pydantic import Field
from typing import Any

class DocumentSearchTool(BaseTool):
    """
    Tool for searching uploaded documents.
    """
    
    # 1. NAME - How the agent refers to this tool
    name: str = "search_documents"
    
    # 2. DESCRIPTION - CRITICAL for agent decision-making
    description: str = """
    Use this tool to search through uploaded PDF documents.
    
    When to use:
    - User asks about content in uploaded documents
    - Need to find specific information from PDFs
    - Looking for quotes, data, or facts from research papers
    
    Input: A search query or question
    Output: Relevant excerpts from documents with source citations
    """
    
    # 3. DEPENDENCIES - Tools can have state
    vectorstore: Any = Field(exclude=True)
    llm: Any = Field(exclude=True)
    
    # 4. IMPLEMENTATION - The actual logic
    def _run(self, query: str) -> str:
        """Execute the tool"""
        # Search the vector store
        docs = self.vectorstore.similarity_search(query, k=3)
        
        # Format results
        output = "Found the following information:\n\n"
        for i, doc in enumerate(docs, 1):
            output += f"[{i}] {doc.page_content}\n"
            output += f"Source: {doc.metadata.get('source', 'Unknown')}\n\n"
        
        return output
    
    async def _arun(self, query: str) -> str:
        """Async version (optional)"""
        return self._run(query)
```

### Tool Description: The Secret Sauce

The **description** is how the agent decides which tool to use. Make it:

✅ **Clear**: State exactly when to use it  
✅ **Specific**: Give examples of use cases  
✅ **Distinctive**: Explain how it differs from other tools  
❌ **Not vague**: Don't just say "searches stuff"

**Example - Bad Description:**
```python
description = "Searches for information"
# Problem: Agent won't know WHEN to use this vs other search tools
```

**Example - Good Description:**
```python
description = """
Use this tool to search the internet for CURRENT information.

When to use:
- User asks about recent events or news
- Need information NOT in uploaded documents
- Questions about events after document publication dates

Input: A search query
Output: Recent web search results with sources
"""
# Clear: Agent knows this is for current/external info
```

---

## Agent Types in LangChain

### 1. Zero-Shot ReAct Description

```python
agent_type = "zero-shot-react-description"
```

**How it works:**
- Reads tool descriptions
- Decides which tool to use based on descriptions
- No prior examples needed ("zero-shot")

**Best for:**
- ✅ Most use cases
- ✅ Custom tools
- ✅ Dynamic tool selection

**Our project uses this!**

### 2. Conversational ReAct Description

```python
agent_type = "conversational-react-description"
```

**How it works:**
- Same as zero-shot, but maintains conversation memory
- Remembers previous questions and answers

**Best for:**
- ✅ Multi-turn conversations
- ✅ Follow-up questions
- ✅ Chatbots

### 3. ReAct Docstore

```python
agent_type = "react-docstore"
```

**How it works:**
- Specialized for document Q&A
- Has built-in "Search" and "Lookup" actions

**Best for:**
- ✅ Document-only applications
- ❌ Limited to document search (can't add custom tools easily)

### Comparison Table

| Agent Type | Memory | Custom Tools | Best Use Case |
|------------|--------|--------------|---------------|
| **zero-shot-react-description** | ❌ No | ✅ Yes | General purpose, custom tools |
| **conversational-react-description** | ✅ Yes | ✅ Yes | Chatbots, multi-turn |
| **react-docstore** | ❌ No | ⚠️ Limited | Document Q&A only |

---

## Building an Agent: Our Implementation

### Step 1: Create Tools

**File: `src/tools/document_search.py`**

```python
class DocumentSearchTool(BaseTool):
    name: str = "search_documents"
    description: str = """
    Use this tool to search through uploaded PDF documents.
    When to use: User asks about content in uploaded documents
    """
    
    vectorstore: Any = Field(exclude=True)
    llm: Any = Field(exclude=True)
    
    def _run(self, query: str) -> str:
        docs = self.vectorstore.similarity_search(query, k=3)
        # Format and return results
        return formatted_results
```

**File: `src/tools/web_search.py`**

```python
class WebSearchTool(BaseTool):
    name: str = "search_web"
    description: str = """
    Use this tool to search the internet for current information.
    When to use: User asks about recent events or news
    """
    
    def _run(self, query: str) -> str:
        # Use Tavily API to search
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=5)
        # Format and return results
        return formatted_results
```

**File: `src/tools/summarization.py`**

```python
class SummarizationTool(BaseTool):
    name: str = "summarize_content"
    description: str = """
    Use this tool to summarize documents.
    When to use: User asks for a summary or overview
    """
    
    llm: Any = Field(exclude=True)
    vectorstore: Any = Field(exclude=True)
    
    def _run(self, instruction: str) -> str:
        # Get all documents
        all_docs = self.vectorstore.get()
        # Use map-reduce to summarize
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        summary = chain.run(all_docs)
        return summary
```

### Step 2: Initialize Agent

**File: `src/agent/research_agent.py`**

```python
from langchain.agents import initialize_agent, AgentType

class ResearchAgent:
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.tools = self._create_default_tools()
    
    def _create_default_tools(self):
        """Create the tools the agent can use"""
        return [
            DocumentSearchTool(
                vectorstore=self.vectorstore,
                llm=self.llm
            ),
            WebSearchTool(),
            SummarizationTool(
                llm=self.llm,
                vectorstore=self.vectorstore
            )
        ]
    
    def create_agent(self, agent_type="zero-shot-react-description", verbose=True):
        """Create the agent executor"""
        
        # Map string to LangChain enum
        agent_type_map = {
            "zero-shot-react-description": AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            "conversational-react-description": AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        }
        
        selected_agent_type = agent_type_map.get(
            agent_type,
            AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
        
        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=selected_agent_type,
            verbose=verbose,  # Shows Thought/Action/Observation
            max_iterations=5,  # Prevent infinite loops
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        
        return self.agent
```

### Step 3: Integrate into Main System

**File: `src/main.py`**

```python
class ResearchAssistant:
    def __init__(self):
        self.vectorstore = None
        self.agent = None
        self.agent_config = AgentConfig()  # Configuration
    
    def setup_agent(self):
        """Initialize the agent"""
        if self.vectorstore is None:
            raise ValueError("Load documents first!")
        
        # Get LLM
        llm = llm_manager.get_llm(
            temperature=self.agent_config.temperature
        )
        
        # Create agent
        research_agent = ResearchAgent(llm, self.vectorstore)
        self.agent = research_agent.create_agent(
            agent_type=self.agent_config.agent_type,
            verbose=self.agent_config.verbose
        )
        
        print("✓ Research agent ready")
        return self.agent
    
    def ask_agent(self, query: str) -> str:
        """Ask the agent a question"""
        if self.agent is None:
            raise ValueError("Call setup_agent() first!")
        
        # Agent autonomously decides which tools to use
        result = self.agent.run(query)
        return result
```

### Step 4: Usage

```python
from src.main import ResearchAssistant

# Initialize
assistant = ResearchAssistant()
assistant.load_documents(["paper.pdf"])
assistant.setup_agent()

# Ask questions - agent decides which tools to use
answer = assistant.ask_agent("What does the paper say about AI?")
# Agent will use: search_documents

answer = assistant.ask_agent("What's new in AI in 2024?")
# Agent will use: search_web

answer = assistant.ask_agent("Summarize all documents")
# Agent will use: summarize_content

answer = assistant.ask_agent("What does the paper say about AI, and what's new in 2024?")
# Agent will use: search_documents, then search_web
```

---

## Agent vs Traditional Chains

### Traditional Chain (Fixed Flow)

```python
# QA Chain - Always retrieves documents
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

result = chain.run("What's new in AI in 2024?")
# Problem: Searches documents (which don't have 2024 info)
# Should search web instead!
```

**Flow:**
```
Question → Retrieve Docs → Generate Answer
```

**Characteristics:**
- ✅ Predictable
- ✅ Fast
- ❌ Inflexible (always same steps)
- ❌ Can't adapt to question type

### Agent (Dynamic Flow)

```python
# Agent - Decides which tool to use
agent = initialize_agent(
    tools=[document_search, web_search, summarize],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

result = agent.run("What's new in AI in 2024?")
# Agent recognizes "2024" = current events
# Chooses: search_web
```

**Flow:**
```
Question → Think → Choose Tool → Execute → Think → Answer
```

**Characteristics:**
- ✅ Flexible (adapts to question)
- ✅ Can use multiple tools
- ✅ Handles complex queries
- ❌ Less predictable
- ❌ Slower (more LLM calls)

### When to Use Each

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| Simple document Q&A | **Chain** | Faster, predictable |
| Current events questions | **Agent** | Needs web search |
| Multi-step research | **Agent** | Can combine tools |
| Production (known queries) | **Chain** | More reliable |
| Exploration/Development | **Agent** | More flexible |
| Cost-sensitive | **Chain** | Fewer LLM calls |

---

## Best Practices

### 1. Write Clear Tool Descriptions

```python
# ❌ Bad
description = "Searches stuff"

# ✅ Good
description = """
Use this tool to search uploaded PDF documents.

When to use:
- User asks about content IN the uploaded documents
- Need specific quotes or data from PDFs

When NOT to use:
- Current events (use search_web instead)
- General knowledge (LLM can answer directly)

Input: Search query
Output: Relevant excerpts with sources
"""
```

### 2. Set Max Iterations

```python
# Prevent infinite loops
agent = initialize_agent(
    tools=tools,
    llm=llm,
    max_iterations=5,  # Stop after 5 tool calls
    early_stopping_method="generate"
)
```

### 3. Handle Parsing Errors

```python
# LLMs sometimes output malformed actions
agent = initialize_agent(
    tools=tools,
    llm=llm,
    handle_parsing_errors=True  # Gracefully handle errors
)
```

### 4. Use Verbose Mode for Debugging

```python
# Development
agent = initialize_agent(tools=tools, llm=llm, verbose=True)
# Shows: Thought → Action → Observation

# Production
agent = initialize_agent(tools=tools, llm=llm, verbose=False)
# Faster, uses fewer tokens
```

### 5. Choose the Right Temperature

```python
# For agents, slightly higher temperature helps with reasoning
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7  # Good balance for agents
)

# Too low (0.0): May be too rigid in tool selection
# Too high (1.0): May make poor decisions
```

### 6. Provide Tool-Specific Error Messages

```python
def _run(self, query: str) -> str:
    try:
        # Tool logic
        return results
    except APIError as e:
        # Return helpful message to agent
        return (
            "API quota exceeded. "
            "Try rephrasing the query or use a different tool."
        )
```

---

## Common Pitfalls

### 1. Vague Tool Descriptions

**Problem:**
```python
description = "Searches for information"
```

Agent can't distinguish between `search_documents` and `search_web`.

**Solution:**
```python
description = """
Search UPLOADED documents (PDFs).
Use for: Questions about paper content
Don't use for: Current events (use search_web)
"""
```

### 2. Too Many Tools

**Problem:**
```python
tools = [tool1, tool2, tool3, ..., tool15]  # 15 tools!
```

Agent gets confused, makes poor choices.

**Solution:**
- Keep it to 3-5 tools
- Combine similar tools
- Use tool categories

### 3. Ignoring Agent Type Parameter

**Problem:**
```python
# In our early code
def create_agent(self, agent_type="zero-shot-react-description"):
    # But then...
    self.agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION  # Hardcoded!
    )
```

The `agent_type` parameter was ignored.

**Solution:**
```python
def create_agent(self, agent_type="zero-shot-react-description"):
    # Map string to enum
    agent_type_map = {
        "zero-shot-react-description": AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        "conversational-react-description": AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    }
    
    selected = agent_type_map.get(agent_type, AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    
    self.agent = initialize_agent(
        agent=selected  # Use the parameter!
    )
```

### 4. Not Handling Rate Limits

**Problem:**
```python
# Agent makes many LLM calls
# Thought (LLM call) → Action (LLM call) → Observation → Thought (LLM call)...
# Quickly hits rate limits!
```

**Solution:**
- Use smaller models for development (`llama-3.1-8b-instant`)
- Turn off verbose mode in production
- Set `max_iterations` to limit calls
- Cache results when possible

### 5. Forgetting to Load Environment Variables

**Problem:**
```python
# In tool
api_key = os.getenv('TAVILY_API_KEY')  # Returns None!
```

**Solution:**
```python
from dotenv import load_dotenv

def _run(self, query: str) -> str:
    load_dotenv()  # Ensure .env is loaded
    api_key = os.getenv('TAVILY_API_KEY')
```

---

## Advanced Topics

### 1. Custom Agent Prompts

You can customize how the agent thinks:

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# Get default ReAct prompt
prompt = hub.pull("hwchase17/react")

# Or create custom prompt
custom_prompt = """
You are a research assistant helping with academic papers.

When answering:
1. Always cite sources
2. Be concise
3. If unsure, say so

Tools available:
{tools}

Use this format:
Thought: [your reasoning]
Action: [tool name]
Action Input: [tool input]
Observation: [tool output]
... (repeat as needed)
Final Answer: [your answer with citations]

Question: {input}
{agent_scratchpad}
"""

agent = create_react_agent(llm, tools, custom_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

### 2. Memory for Conversational Agents

```python
from langchain.memory import ConversationBufferMemory

# Add memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,  # Agent remembers conversation
    verbose=True
)

# Now agent can handle follow-ups
agent.run("What does the paper say about AI?")
agent.run("Can you elaborate on that?")  # Remembers previous answer
```

### 3. Structured Tools with Pydantic

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Number of results")

def search_function(query: str, max_results: int = 5) -> str:
    # Search logic
    return results

search_tool = StructuredTool.from_function(
    func=search_function,
    name="search_documents",
    description="Search uploaded documents",
    args_schema=SearchInput
)
```

### 4. Async Agents for Better Performance

```python
async def run_agent_async(query: str):
    result = await agent.arun(query)
    return result

# Run multiple queries concurrently
import asyncio

queries = [
    "What is AI?",
    "What is machine learning?",
    "What is deep learning?"
]

results = await asyncio.gather(*[
    run_agent_async(q) for q in queries
])
```

### 5. Agent Callbacks for Monitoring

```python
from langchain.callbacks import StdOutCallbackHandler

class CustomCallback(StdOutCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"🔧 Using tool: {serialized['name']}")
    
    def on_tool_end(self, output, **kwargs):
        print(f"✅ Tool finished: {output[:100]}...")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    callbacks=[CustomCallback()],
    verbose=True
)
```

---

## Summary: Key Takeaways

### What You Learned

1. **Agents** = LLM + Tools + Reasoning Loop
2. **ReAct Pattern** = Thought → Action → Observation → Repeat
3. **Tool Descriptions** are critical for agent decision-making
4. **Agent Types**: Zero-shot (general), Conversational (memory), Docstore (docs only)
5. **Agents vs Chains**: Agents are flexible but slower; chains are fast but rigid

### Our Implementation

```
ResearchAssistant
    ├── DocumentSearchTool (search uploaded PDFs)
    ├── WebSearchTool (search internet with Tavily)
    └── SummarizationTool (summarize documents)

Agent Type: zero-shot-react-description
Pattern: ReAct (Reasoning + Acting)
LLM: Groq (llama-3.1-8b-instant)
```

### When to Use Agents

✅ **Use agents when:**
- Need to combine multiple data sources
- Questions require different approaches
- Want autonomous decision-making
- Handling complex, multi-step tasks

❌ **Don't use agents when:**
- Simple, predictable queries
- Speed is critical
- Cost is a major concern
- Need deterministic behavior

### Next Steps

1. **Experiment**: Try different agent types
2. **Customize**: Write custom prompts
3. **Extend**: Add more tools (calculator, database, API calls)
4. **Optimize**: Fine-tune tool descriptions
5. **Monitor**: Add callbacks to track agent behavior

---

## Resources

- [LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Our Implementation](file:///c:/Users/kissa/OneDrive/Desktop/research-assistant/src/agent/)
- [Tool Examples](file:///c:/Users/kissa/OneDrive/Desktop/research-assistant/src/tools/)

---

**Happy Building! 🚀**
