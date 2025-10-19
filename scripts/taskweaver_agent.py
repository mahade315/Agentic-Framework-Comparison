# taskweaver_agent.py
"""
TaskWeaver agent implementation for HumanEval code generation.
Minimal implementation following the same interface as other agents.
"""

import os
from dotenv import load_dotenv
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

def generate_one_completion(prompt: str) -> str:
    """
    Generate a single completion using TaskWeaver-style agent approach.
    TaskWeaver is a code-first agent framework for data analytics tasks.
    """
    try:
        # TaskWeaver typically uses a code-first approach with planning and execution
        # For HumanEval, we'll simulate the agent's code generation process
        
        # Step 1: TaskWeaver agent plans the approach
        planning_prompt = f"""
        As a TaskWeaver agent, plan and execute this HumanEval task:
        
        {prompt}
        
        Follow TaskWeaver's code-first approach:
        1. Analyze the problem requirements
        2. Plan the solution approach
        3. Generate the implementation code
        
        Return ONLY the function body with proper indentation.
        """
        
        # For now, we'll use a direct approach similar to OpenAI
        # In a full TaskWeaver implementation, this would go through planning and execution phases
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a TaskWeaver agent specialized in code-first data analytics. "
                        "Plan and execute the solution, generating ONLY the function body with proper indentation. "
                        "Follow a systematic approach: analyze, plan, implement. "
                        "No explanations, just the correct Python code."
                    ),
                },
                {
                    "role": "user",
                    "content": planning_prompt,
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        
        # Track token usage
        if hasattr(response, 'usage') and response.usage:
            token_usage["input_tokens"] += response.usage.prompt_tokens
            token_usage["output_tokens"] += response.usage.completion_tokens
            token_usage["total_tokens"] += response.usage.total_tokens
        
        raw_output = response.choices[0].message.content or ""
        
        # TaskWeaver agents typically have structured output with code blocks
        # Extract just the function body
        lines = raw_output.strip().split('\n')
        function_lines = []
        in_code_block = False
        in_function = False
        
        for line in lines:
            # Look for code blocks (```python or ```)
            if line.strip().startswith('```python') or line.strip().startswith('```'):
                in_code_block = True
                continue
            elif in_code_block and line.strip().startswith('```'):
                in_code_block = False
                continue
            
            if in_code_block or in_function:
                if line.strip().startswith('def ') or in_function:
                    in_function = True
                    if line.strip().startswith('def '):
                        # Skip the function signature, we only want the body
                        continue
                    function_lines.append(line)
        
        if function_lines:
            raw_output = '\n'.join(function_lines)
        else:
            # Fallback to original output
            raw_output = raw_output.strip()
        
        return sanitize_completion(raw_output)
        
    except Exception as e:
        print(f"\n❌ TaskWeaver agent error: {e}")
        return "    pass  # Error generating completion"

def reset_agent():
    """Reset agent state (placeholder for future use)."""
    pass
