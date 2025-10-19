"""
Utilities to sanitize model outputs for HumanEval.

We aim to return ONLY the function *body* (suffix) for a given prompt.
- Strip markdown code fences.
- If the model returned the full function, drop the `def ...:` line and dedent.
- Remove trailing prose or stray fences.
"""

import re


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    # Prefer fenced python blocks if present
    fenced = re.findall(r"```(?:python)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    # Generic triple-backtick wrapper
    if text.startswith("```") and text.endswith("```"):
        return text[3:-3].strip()
    return text


def _strip_signature_if_present(text: str) -> str:
    """
    If the model returned a full function (signature + body),
    drop the first 'def ...:' line but keep the body indented.
    """
    lines = text.splitlines()
    if not lines:
        return text

    # Match a def line with trailing colon.
    if re.match(r"^\s*def\s+\w+\s*\(.*\)\s*:\s*$", lines[0]):
        body_lines = lines[1:]
        text = "\n".join(body_lines).lstrip("\n")
    else:
        # If no signature found, ensure proper indentation (4 spaces)
        # Check if the code is already indented
        non_empty = [ln for ln in lines if ln.strip()]
        if non_empty:
            min_indent = min(len(re.match(r"^[ \t]*", ln).group(0)) for ln in non_empty)
            if min_indent < 4:
                # Add 4 spaces indentation to all non-empty lines
                lines = ["    " + ln if ln.strip() else ln for ln in lines]
                text = "\n".join(lines)

    # Ensure trailing newline for many HumanEval tasks
    return text if text.endswith("\n") else (text + "\n")


def sanitize_completion(text: str) -> str:
    """Compose all sanitizers to produce a clean suffix."""
    text = _strip_code_fences(text)

    # prune any trailing fenced code or markdown artifacts
    lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            break
        lines.append(ln)
    text = "\n".join(lines).rstrip() + "\n"

    text = _strip_signature_if_present(text)

    # Finally, avoid triple backticks that slipped through
    text = text.replace("```", "").rstrip() + "\n"
    return text