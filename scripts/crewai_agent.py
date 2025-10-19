# crewai_agent_optimized.py
"""
Ultra-optimized CrewAI agent that mimics direct API performance.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from sanitize import sanitize_completion

load_dotenv()

# Configuration
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
VERBOSE = os.getenv("CREWAI_VERBOSE", "false").lower() == "true"

# Singleton instances
_agent_instance = None
_llm_instance = None

# Global token tracking
token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def get_token_usage():
    """Get current token usage statistics."""
    return token_usage.copy()

def reset_token_usage():
    """Reset token usage statistics."""
    global token_usage
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _get_llm():
    """Get or create LLM instance (singleton)."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _llm_instance


def _get_code_agent():
    """Get ultra-minimal agent (singleton)."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = Agent(
            role="Code Generator",
            goal="Write function bodies",
            backstory="Write Python function bodies only.",
            llm=_get_llm(),
            verbose=False,  # Always silent
            allow_delegation=False,
            max_iter=1,
        )
    return _agent_instance


def generate_one_completion(prompt: str) -> str:
    """
    Ultra-optimized CrewAI completion that mimics direct API.
    """
    try:
        agent = _get_code_agent()
        
        # Minimal task - closest to direct API prompt
        task = Task(
            description=f"{prompt}\n\n# Write ONLY the function body below, nothing else.",
            expected_output="Function body",
            agent=agent,
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,  # Always silent
        )
        
        result = crew.kickoff()
        
        # Extract string
        if hasattr(result, 'raw'):
            raw_output = result.raw
        elif isinstance(result, str):
            raw_output = result
        else:
            raw_output = str(result)
        
        # Estimate token usage (rough approximation)
        # CrewAI typically uses more tokens due to agent overhead
        estimated_input = len(prompt.split()) * 1.3  # Rough token estimation
        estimated_output = len(raw_output.split()) * 1.3
        
        token_usage["input_tokens"] += int(estimated_input)
        token_usage["output_tokens"] += int(estimated_output)
        token_usage["total_tokens"] += int(estimated_input + estimated_output)
        
        return sanitize_completion(raw_output)
        
    except Exception as e:
        print(f"\n❌ CrewAI error: {e}")
        return "    pass\n"


def reset_agent():
    """Reset singleton instances."""
    global _agent_instance, _llm_instance
    _agent_instance = None
    _llm_instance = None

