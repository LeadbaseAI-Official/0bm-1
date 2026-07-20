import re
import json
import pickle
from pathlib import Path
from model import get_llm, STATES_DIR
from chat_template import format_chat_prompt

def run_warmup() -> None:
    print("[Warmup] Starting standalone prefix warmup...", flush=True)
    system_path = Path("system.md")
    persona_path = Path("persona.json")
    kb_path = Path("kb.json")
    
    if system_path.exists() and persona_path.exists() and kb_path.exists():
        system_prompt = system_path.read_text(encoding="utf-8").strip()
        persona_raw = persona_path.read_text(encoding="utf-8").strip()
        kb = kb_path.read_text(encoding="utf-8").strip()
        
        # Strip tool declarations
        clean_system = re.sub(r"## TOOL USE[\s\S]*?(?=##|$)", "", system_prompt)
        
        # Format persona exactly like auto-reply.ts
        try:
            persona_data = json.loads(persona_raw)
            if isinstance(persona_data.get("persona"), dict):
                persona = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in persona_data["persona"].items())
            else:
                persona = str(persona_data.get("persona", "none"))
        except Exception:
            persona = "none"
        
        prompt_parts = [
            "System Prompt:",
            clean_system,
            "",
            "Persona:",
            persona,
            "",
            "Knowledge Base (Authoritative Facts):",
            kb,
            "",
            "Conversation History:",
            "none",
            "",
            "Current Query:",
            "Hi",
            "",
            "Instruction: Answer the customer's query using only the facts in the Knowledge Base above. Be helpful, polite, and professional.",
            "",
            "Assistant:"
        ]
        prompt_text = "\n".join(prompt_parts)
        formatted_prefix = format_chat_prompt(prompt_text)
        
        llm = get_llm()
        prefix_tokens = llm.tokenize(formatted_prefix.encode("utf-8"))
        
        state_file = STATES_DIR / "global_prefix.state"
        
        print(f"[Warmup] Compiling global prefix cache ({len(prefix_tokens)} tokens)...", flush=True)
        llm.reset()
        llm.eval(prefix_tokens)
        
        state_data = llm.save_state()
        with open(state_file, "wb") as sf:
            pickle.dump({"state": state_data, "tokens": prefix_tokens}, sf)
        print(f"[Warmup] Warmup complete. Saved state to {state_file}.", flush=True)
    else:
        print("[Warmup] Error: Required warmup files not found.", flush=True)

if __name__ == "__main__":
    run_warmup()
