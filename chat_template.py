def format_system_prompt(system_prompt: str) -> str:
    """Formats system prompt block (SYSTEM PROMPT + KB + PERSONA) for Qwen ChatML template."""
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

def format_user_turn(user_message: str) -> str:
    """Formats user query turn header for Qwen ChatML template."""
    return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

def format_chat_prompt(prompt: str, system_prompt: str = "") -> str:
    """Formats full standalone chat prompt with optional system prompt in Qwen ChatML structure."""
    if system_prompt:
        return f"{format_system_prompt(system_prompt)}{format_user_turn(prompt)}"
    return format_user_turn(prompt)

