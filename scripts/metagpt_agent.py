# metagpt_agent.py
"""
MetaGPT agent implementation for HumanEval code generation.
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
    Generate a single completion using MetaGPT-style agent approach.
    This is a simplified implementation that mimics MetaGPT's agent behavior.
    """
    try:
        # MetaGPT typically uses a multi-step agent approach
        # For HumanEval, we'll simulate the agent's code generation process
        
        # Step 1: Agent analyzes the problem (simulated)
        analysis_prompt = f"""
        As a MetaGPT agent, analyze this HumanEval problem:
        
        {prompt}
        
        Identify:
        1. The function signature and requirements
        2. The expected behavior from examples
        3. The approach needed to solve it
        
        Then generate ONLY the function body with proper indentation.
        """
        
        # For now, we'll use a direct approach similar to OpenAI
        # In a full MetaGPT implementation, this would go through multiple agent steps
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a MetaGPT agent specialized in Python code generation. "
                        "Analyze the problem and generate ONLY the function body with proper indentation. "
                        "No explanations, no reasoning, just the correct Python code."
                    ),
                },
                {
                    "role": "user",
                    "content": analysis_prompt,
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
        
        # MetaGPT agents typically have more structured output
        # Extract just the function body
        lines = raw_output.strip().split('\n')
        function_lines = []
        in_function = False
        
        for line in lines:
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
        print(f"\n❌ MetaGPT agent error: {e}")
        return "    pass  # Error generating completion"

def reset_agent():
    """Reset agent state (placeholder for future use)."""
    pass
