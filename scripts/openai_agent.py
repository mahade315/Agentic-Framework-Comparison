# openai_agent.py
"""
OpenAI Agents SDK implementation for HumanEval code generation.
Uses OpenAI Agents SDK with OpenAI models for direct agent-based code generation.
Similar structure to CrewAI, Qwen-Agent, LangChain, and LangGraph: Direct Response (no tools)
"""

import os
from dotenv import load_dotenv
from agents import Agent, Runner
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

def _get_openai_agent():
    """Get or create OpenAI Agent instance (singleton)."""
    global _agent_instance
    
    if _agent_instance is None:
        # Create OpenAI Agent with NO TOOLS (no tools parameter = no tools)
        _agent_instance = Agent(
            name="Code Generator",
            instructions="""You are a Python code generator. When given a HumanEval problem, generate only the function body code. Do not include the function signature, docstring, or any explanations. Just return the indented function body.""",
            model=MODEL,
            # No tools parameter = no tools
        )
    
    return _agent_instance

def generate_one_completion(prompt: str) -> str:
    """
    Generate a single completion using OpenAI Agent.
    Direct response approach - similar to CrewAI, Qwen-Agent, LangChain, and LangGraph (no tools).
    """
    try:
        # Get the OpenAI Agent
        agent = _get_openai_agent()
        
        # Create a focused prompt for the agent
        agent_input = f"""Generate the function body for this HumanEval problem:

{prompt}

Generate only the function body code, no explanations or markdown."""
        
        # Run the agent synchronously
        result = Runner.run_sync(agent, agent_input)
        
        # Extract the final output
        if result and hasattr(result, 'final_output'):
            content = result.final_output
            
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
            
            # Estimate token usage (since OpenAI Agents doesn't provide exact counts)
            estimated_input = len(agent_input.split()) * 1.3
            estimated_output = len(code.split()) * 1.3
            
            token_usage["input_tokens"] += int(estimated_input)
            token_usage["output_tokens"] += int(estimated_output)
            token_usage["total_tokens"] += int(estimated_input + estimated_output)
            
            return sanitize_completion(code)
        
        return "    pass  # No response generated"
        
    except Exception as e:
        print(f"❌ OpenAI Agent error: {e}")
        return "    pass  # Error generating completion"

def reset_agent():
    """Reset singleton instances."""
    global _agent_instance
    _agent_instance = None
