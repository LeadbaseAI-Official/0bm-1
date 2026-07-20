from pathlib import Path
from typing import Optional, Dict, Any
from llama_cpp import Llama, GGML_TYPE_Q8_0
from chat_template import format_chat_prompt

MODEL_CODE = "0bm"

_llm_instance: Optional[Llama] = None
_states: Dict[str, Any] = {}

def find_gguf_file() -> Path:
    # Check current directory
    for path in Path(".").glob("*.gguf"):
        # Make sure it's not the mmproj file
        if "mmproj" not in path.name:
            return path
    # Check model/ directory
    model_dir: Path = Path("model")
    if model_dir.exists():
        for path in model_dir.glob("*.gguf"):
            if "mmproj" not in path.name:
                return path
    # Fallback default
    return Path("Qwen3.5-0.8B-Q4_K_M.gguf")

def find_mmproj_file() -> Optional[Path]:
    for path in Path(".").glob("*mmproj*.gguf"):
        return path
    model_dir: Path = Path("model")
    if model_dir.exists():
        for path in model_dir.glob("*mmproj*.gguf"):
            return path
    return None

def get_llm() -> Llama:
    global _llm_instance
    if _llm_instance is None:
        model_path: Path = find_gguf_file()
        if not model_path.exists():
            raise FileNotFoundError(f"No GGUF model file found. Expected one in root or model/ directory.")
        
        mmproj_path = find_mmproj_file()
        chat_handler = None
        if mmproj_path:
            try:
                from llama_cpp.llama_chat_format import LlavaChatHandler
                print(f"[Model] Found vision projector file: {mmproj_path}", flush=True)
                chat_handler = LlavaChatHandler(clip_model_path=str(mmproj_path))
            except Exception as e:
                print(f"[Model] Warning: Failed to load LlavaChatHandler: {e}", flush=True)
        
        # Optimize for 2-core GitHub Action CPU runners: n_threads=2, n_ctx=40960 (limits state size to ~60MB)
        _llm_instance = Llama(
            model_path=str(model_path),
            n_threads=2,
            n_ctx=40960,
            flash_attn=True,
            type_k=GGML_TYPE_Q8_0,
            type_v=GGML_TYPE_Q8_0,
            chat_handler=chat_handler
        )
    return _llm_instance

import pickle
import threading
import os
import asyncio

# Create states directory
STATES_DIR = Path("states")
STATES_DIR.mkdir(parents=True, exist_ok=True)

_llm_lock = asyncio.Lock()

def save_state_bg(state_file: Path, state_obj: Any, tokens: list) -> None:
    try:
        tmp_file = state_file.with_suffix(f".{threading.get_ident()}.tmp")
        with open(tmp_file, "wb") as sf:
            pickle.dump({"state": state_obj, "tokens": tokens}, sf)
        os.replace(tmp_file, state_file)
        print(f"[Model] Background state saved to {state_file.name}", flush=True)
    except Exception as e:
        print(f"[Model] Background state save warning: {e}", flush=True)

async def run_model_query(prompt: str, jid: Optional[str] = None, image_base64: Optional[str] = None) -> str:
    async with _llm_lock:
        def evaluate_query() -> str:
            nonlocal prompt, image_base64
            try:
                llm: Llama = get_llm()
                
                if image_base64 and getattr(llm, "chat_handler", None) is not None:
                    print(f"[Model] Running vision query with image of size {len(image_base64)} characters", flush=True)
                    if not image_base64.startswith("data:image"):
                        image_base64 = f"data:image/jpeg;base64,{image_base64}"
                    
                    # Apply logit_bias to ban <|channel>thought token generation
                    logit_bias = {}
                    try:
                        thought_token_id = llm.tokenize(b"<|channel>thought")[-1]
                        logit_bias[thought_token_id] = -100.0
                    except Exception:
                        pass

                    response_generator = llm.create_chat_completion(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": image_base64}}
                                ]
                            }
                        ],
                        max_tokens=512,
                        stream=True,
                        logit_bias=logit_bias
                    )
                    text_chunks = []
                    print("[Model Vision] Generating: ", end="", flush=True)
                    for chunk in response_generator:
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            token_text = delta["content"]
                            print(token_text, end="", flush=True)
                            text_chunks.append(token_text)
                    print("\n[Model Vision] Generation complete.", flush=True)
                    text_result = "".join(text_chunks)
                else:
                    if image_base64:
                        print(f"[Model] Text fallback mode: Received image of size {len(image_base64)} characters", flush=True)
                        prompt = f"[User uploaded an image. Base64 length: {len(image_base64)}]\n{prompt}"
                    
                    formatted_prompt: str = format_chat_prompt(prompt)
                    new_tokens = llm.tokenize(formatted_prompt.encode("utf-8"))
                    print(f"[Model] Received query prompt. Token count: {len(new_tokens)} tokens.", flush=True)
                    
                    state_file = STATES_DIR / f"{jid}.state" if jid else None
                    common_len = 0
                    
                    # 1. First, try loading the specific JID cache state
                    if jid and state_file.exists():
                        try:
                            with open(state_file, "rb") as sf:
                                raw_data = pickle.load(sf)
                            cached_tokens = raw_data["tokens"]
                            
                            for t1, t2 in zip(cached_tokens, new_tokens):
                                if t1 != t2:
                                    break
                                common_len += 1
                                
                            if common_len > 0:
                                llm.load_state(raw_data["state"])
                                print(f"[Model] Restored cache state for JID: {jid} (prefix matched: {common_len}/{len(cached_tokens)} tokens)", flush=True)
                            else:
                                print(f"[Model] Cache files exist for JID {jid} but matched 0 tokens. Cache mismatch.", flush=True)
                        except Exception as cache_err:
                            print(f"[Model] Warning: Failed to load cache state for JID {jid}: {cache_err}. Deleting invalid cache.", flush=True)
                            try:
                                state_file.unlink()
                            except Exception:
                                pass
                            common_len = 0
                    else:
                        if jid:
                            print(f"[Model] No specific cache files found for JID: {jid}", flush=True)
                    
                    # 2. If JID cache missed, fall back to the pre-cached global prefix
                    if common_len == 0:
                        global_state = STATES_DIR / "global_prefix.state"
                        if global_state.exists():
                            try:
                                with open(global_state, "rb") as sf:
                                    raw_data = pickle.load(sf)
                                cached_tokens = raw_data["tokens"]
                                
                                for t1, t2 in zip(cached_tokens, new_tokens):
                                    if t1 != t2:
                                        break
                                    common_len += 1
                                    
                                if common_len > 0:
                                    llm.load_state(raw_data["state"])
                                    print(f"[Model] Restored cache from global prefix cache (prefix matched: {common_len}/{len(cached_tokens)} tokens)", flush=True)
                                    if common_len < len(cached_tokens):
                                        print(f"[Model] Cache mismatch at token {common_len}!", flush=True)
                                        try:
                                            matched_part = llm.detokenize(cached_tokens[max(0, common_len-10):common_len]).decode("utf-8", errors="replace")
                                            cached_diff = llm.detokenize(cached_tokens[common_len:common_len+20]).decode("utf-8", errors="replace")
                                            new_diff = llm.detokenize(new_tokens[common_len:common_len+20]).decode("utf-8", errors="replace")
                                            print(f"[Model] Last matched: {repr(matched_part)}", flush=True)
                                            print(f"[Model] Cached expected:  {repr(cached_diff)}", flush=True)
                                            print(f"[Model] New query got:    {repr(new_diff)}", flush=True)
                                        except Exception as diff_err:
                                            print(f"[Model] Failed to print mismatch details: {diff_err}", flush=True)
                                else:
                                    print(f"[Model] Global prefix cache exists but matched 0 tokens against current prompt.", flush=True)
                            except Exception as glob_err:
                                print(f"[Model] Warning: Failed to load global prefix cache: {glob_err}. Rebuilding state...", flush=True)
                                try:
                                    global_state.unlink()
                                except Exception:
                                    pass
                                common_len = 0
                        else:
                            print(f"[Model] No pre-cached global prefix files found at {global_state}", flush=True)
                                
                    if common_len == 0:
                        llm.reset()
                        print(f"[Model] Cache miss/fresh start for JID: {jid}. Prompt must be evaluated from scratch.", flush=True)
                    
                    # Apply logit_bias to ban <|channel>thought token generation
                    logit_bias = {}
                    try:
                        thought_token_id = llm.tokenize(b"<|channel>thought")[-1]
                        logit_bias[thought_token_id] = -100.0
                    except Exception:
                        pass

                    response_generator = llm(
                        formatted_prompt,
                        max_tokens=512,
                        stream=True,
                        logit_bias=logit_bias
                    )
                    
                    text_result_chunks = []
                    print("[Model] Generating: ", end="", flush=True)
                    for chunk in response_generator:
                        token_text = chunk["choices"][0]["text"]
                        print(token_text, end="", flush=True)
                        text_result_chunks.append(token_text)
                    print("\n[Model] Generation complete.", flush=True)
                    text_result = "".join(text_result_chunks)
                    
                    if jid:
                        try:
                            state_data = llm.save_state()
                            full_evaluated_text = formatted_prompt + text_result
                            full_tokens = llm.tokenize(full_evaluated_text.encode("utf-8"))
                            
                            t = threading.Thread(
                                target=save_state_bg,
                                args=(state_file, state_data, full_tokens),
                                daemon=True
                            )
                            t.start()
                        except Exception as save_err:
                            print(f"[Model] Warning: Failed to save cache state for JID {jid}: {save_err}", flush=True)
                            import traceback
                            traceback.print_exc()
                    
                return text_result
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"Exception raised while running llama-cpp: {e}"

        return await asyncio.to_thread(evaluate_query)