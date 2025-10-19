# qwen_agent.py
"""
Qwen-Agent implementation for HumanEval code generation.
Uses Qwen-Agent framework with OpenAI models for direct agent-based code generation.
Similar structure to CrewAI: Agent + Direct Response
"""

import os
from dotenv import load_dotenv
from qwen_agent.agents import Assistant
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

def _get_code_agent():
    """Get or create Qwen-Agent instance (singleton)."""
    global _agent_instance
    
    if _agent_instance is None:
        # Configure LLM to use OpenAI API
        llm_cfg = {
            'model': MODEL,
            'model_server': 'https://api.openai.com/v1',  # OpenAI API endpoint
            'api_key': os.getenv('OPENAI_API_KEY'),
            'generate_cfg': {
                'temperature': TEMPERATURE,
                'max_tokens': MAX_TOKENS,
            }
        }
        
        # Create agent for direct code generation (no tools)
        _agent_instance = Assistant(
            llm=llm_cfg,
            system_message=(
                "You are a Python code generator. When given a HumanEval problem, "
                "generate only the function body code. Do not include the function signature, "
                "docstring, or any explanations. Just return the indented function body."
            )
        )
    
    return _agent_instance

def generate_one_completion(prompt: str) -> str:
    """
    Generate a single completion using Qwen-Agent framework.
    Direct agent response approach (no tools) - similar to CrewAI.
    """
    try:
        # Get the Qwen-Agent
        agent = _get_code_agent()
        
        # Create a focused prompt for the agent
        agent_prompt = f"""Generate the function body for this HumanEval problem:

{prompt}

Generate only the function body code, no explanations or markdown."""
        
        # Create messages for the agent
        messages = [{
            'role': 'user', 
            'content': agent_prompt
        }]
        
        # Run the agent
        response = []
        for chunk in agent.run(messages=messages):
            response.extend(chunk)
        
        # Extract the final response
        if response:
            # Get the last assistant message
            assistant_messages = [msg for msg in response if msg.get('role') == 'assistant']
            if assistant_messages:
                content = assistant_messages[-1].get('content', '')
                
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
                
                # Estimate token usage (since Qwen-Agent doesn't provide exact counts)
                estimated_input = len(agent_prompt.split()) * 1.3
                estimated_output = len(code.split()) * 1.3
                
                token_usage["input_tokens"] += int(estimated_input)
                token_usage["output_tokens"] += int(estimated_output)
                token_usage["total_tokens"] += int(estimated_input + estimated_output)
                
                return sanitize_completion(code)
        
        return "    pass  # No response generated"
        
    except Exception as e:
        print(f"❌ Qwen-Agent error: {e}")
        return "    pass  # Error generating completion"

def reset_agent():
    """Reset singleton instances."""
    global _agent_instance
    _agent_instance = None
