# langchain_agent.py
"""
LangChain Agent Executor implementation for HumanEval code generation.
Uses LangChain Agent Executor with OpenAI models for direct agent-based code generation.
Similar structure to CrewAI and Qwen-Agent: Agent + Direct Response (no tools)
"""

import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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
_agent_executor = None
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

def _get_agent_executor():
    """Get or create LangChain Agent Executor instance (singleton)."""
    global _agent_executor
    
    if _agent_executor is None:
        # Create the LLM
        llm = _get_llm()
        
        # Create a simple prompt template for direct code generation
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a Python code generator. When given a HumanEval problem, generate only the function body code. Do not include the function signature, docstring, or any explanations. Just return the indented function body."""),
            ("user", "{input}")
        ])
        
        # Create a simple agent using the LLM directly (no tools)
        # We'll create a custom agent executor that just uses the LLM
        class SimpleAgentExecutor:
            def __init__(self, llm, prompt_template):
                self.llm = llm
                self.prompt_template = prompt_template
            
            def invoke(self, inputs):
                # Create the chain
                chain = self.prompt_template | self.llm
                # Generate response
                response = chain.invoke(inputs)
                return {"output": response.content if hasattr(response, 'content') else str(response)}
        
        _agent_executor = SimpleAgentExecutor(llm, prompt_template)
    
    return _agent_executor

def generate_one_completion(prompt: str) -> str:
    """
    Generate a single completion using LangChain Agent Executor.
    Agent Executor approach with NO TOOLS - similar to CrewAI and Qwen-Agent.
    """
    try:
        # Get the agent executor
        agent_executor = _get_agent_executor()
        
        # Create a focused prompt for the agent
        agent_input = f"""Generate the function body for this HumanEval problem:

{prompt}

Generate only the function body code, no explanations or markdown."""
        
        # Run the agent
        result = agent_executor.invoke({"input": agent_input})
        
        # Extract the response
        if result and "output" in result:
            response = result["output"]
            
            # Extract code from the response
            if '```python' in response:
                # Extract code from markdown
                start = response.find('```python') + 9
                end = response.find('```', start)
                if end > start:
                    code = response[start:end].strip()
                else:
                    code = response[start:].strip()
            else:
                # Use response directly
                code = response
            
            # Estimate token usage (since LangChain doesn't provide exact counts)
            estimated_input = len(agent_input.split()) * 1.3
            estimated_output = len(code.split()) * 1.3
            
            token_usage["input_tokens"] += int(estimated_input)
            token_usage["output_tokens"] += int(estimated_output)
            token_usage["total_tokens"] += int(estimated_input + estimated_output)
            
            return sanitize_completion(code)
        
        return "    pass  # No response generated"
        
    except Exception as e:
        print(f"❌ LangChain Agent error: {e}")
        return "    pass  # Error generating completion"

def reset_agent():
    """Reset singleton instances."""
    global _agent_executor, _llm_instance
    _agent_executor = None
    _llm_instance = None