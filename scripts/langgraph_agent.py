# langgraph_agent.py
"""
LangGraph implementation for HumanEval code generation.
Uses LangGraph framework with OpenAI models for direct agent-based code generation.
Similar structure to CrewAI, Qwen-Agent, and LangChain: Direct Response (no tools)
"""

import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from sanitize import sanitize_completion

load_dotenv()

# Configuration
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "512"))

# Global token tracking
token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def get_token_usage():
    """Get current token usage statistics."""
    return token_usage.copy()

def reset_token_usage():
    """Reset token usage statistics."""
    global token_usage
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

# Singleton instances
_agent_instance = None
_llm_instance = None

def _get_llm():
    """Get or create OpenAI LLM instance (singleton)."""
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    return _llm_instance

def _get_langgraph_agent():
    """Get or create LangGraph agent instance (singleton)."""
    global _agent_instance
    
    if _agent_instance is None:
        # Create the LLM
        llm = _get_llm()
        
        # Create LangGraph agent with NO TOOLS (empty list)
        _agent_instance = create_react_agent(
            model=llm,
            tools=[],  # No tools - empty list
            prompt="""You are a Python code generator. When given a HumanEval problem, generate only the function body code. Do not include the function signature, docstring, or any explanations. Just return the indented function body."""
        )
    
    return _agent_instance

def generate_one_completion(prompt: str) -> str:
    """
    Generate a single completion using LangGraph agent.
    Direct response approach - similar to CrewAI, Qwen-Agent, and LangChain (no tools).
    """
    try:
        # Get the LangGraph agent
        agent = _get_langgraph_agent()
        
        # Create a focused prompt for the agent
        agent_input = f"""Generate the function body for this HumanEval problem:

{prompt}

Generate only the function body code, no explanations or markdown."""
        
        # Run the agent
        result = agent.invoke({
            "messages": [{"role": "user", "content": agent_input}]
        })
        
        # Extract the response from the messages
        if result and "messages" in result:
            messages = result["messages"]
            if messages:
                # Get the last assistant message
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    content = last_message.content
                else:
                    content = str(last_message)
                
                # Extract code from the response
                if '```python' in content:
                    # Extract code from markdown
                    start = content.find('```python') + 9
                    end = content.find('```', start)
                    if end > start:
                        code = content[start:end].strip()
                    else:
                        code = content[start:].strip()
                else:
                    # Use content directly
                    code = content
                
                # Estimate token usage (since LangGraph doesn't provide exact counts)
                estimated_input = len(agent_input.split()) * 1.3
                estimated_output = len(code.split()) * 1.3
                
                token_usage["input_tokens"] += int(estimated_input)
                token_usage["output_tokens"] += int(estimated_output)
                token_usage["total_tokens"] += int(estimated_input + estimated_output)
                
                return sanitize_completion(code)
        
        return "    pass  # No response generated"
        
    except Exception as e:
        print(f"❌ LangGraph error: {e}")
        return "    pass  # Error generating completion"

def reset_agent():
    """Reset singleton instances."""
    global _agent_instance, _llm_instance
    _agent_instance = None
    _llm_instance = None
