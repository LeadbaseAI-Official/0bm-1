def format_chat_prompt(prompt: str, system_prompt: str = "") -> str:
    if system_prompt:
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
