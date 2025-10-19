# OpenAI_models.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from sanitize import sanitize_completion

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "300"))

_client = OpenAI()

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
    Given a HumanEval prompt (signature + docstring), return ONLY the function body.
    Simple Chat Completions call, no retries. Sanitization is handled in sanitize.py.
    """
    resp = _client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You complete Python functions from a provided signature+docstring. "
                    "Return ONLY the function body (the indented code after the signature). "
                    "Do not repeat the signature. Do not add imports. "
                    "Do not include explanations or markdown."
                ),
            },
            {
                "role": "user",
                "content": prompt + "\n\n# Write ONLY the function body below, nothing else.",
            },
        ],
        stop=[
            "\n\n\n",
            "\nif __name__ == '__main__':",
            "\nif __name__ == \"__main__\":",
        ],
    )
    
    # Track token usage
    if hasattr(resp, 'usage') and resp.usage:
        token_usage["input_tokens"] += resp.usage.prompt_tokens
        token_usage["output_tokens"] += resp.usage.completion_tokens
        token_usage["total_tokens"] += resp.usage.total_tokens
    
    raw = (resp.choices[0].message.content or "").strip()
    return sanitize_completion(raw)