import re

def clean_response(raw_text: str, system_prompt: str) -> str:

    if raw_text.startswith(system_prompt):
        cleaned = raw_text[len(system_prompt):].strip()
        cleaned = re.sub(r'^[\.\:\-\s]+', '', cleaned)
        return cleaned
    return raw_text