from pathlib import Path
from typing import Optional, Dict, Any
from llama_cpp import Llama, GGML_TYPE_Q8_0
from chat_template import format_chat_prompt
import pickle
import threading
import os
import asyncio

# Create states directory
STATES_DIR = Path("states")
STATES_DIR.mkdir(parents=True, exist_ok=True)

_llm_lock = asyncio.Lock()
_llm_instance: Optional[Llama] = None
_states: Dict[str, Any] = {}


MODEL_CODE = "0bm"


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


def save_state_bg(state_file: Path, state_obj: Any, tokens: list) -> None:
    try:
        tmp_file = state_file.with_suffix(f".{threading.get_ident()}.tmp")
        with open(tmp_file, "wb") as sf:
            pickle.dump({"state": state_obj, "tokens": tokens}, sf)
        os.replace(tmp_file, state_file)
        print(f"[Model] Background state saved to {state_file.name}", flush=True)
    except Exception as e:
        print(f"[Model] Background state save warning: {e}", flush=True)

async def run_model_query(prompt: str, client_id: Optional[str] = None, phone_number: Optional[str] = None, image_base64: Optional[str] = None) -> str:
    async with _llm_lock:
        def evaluate_query() -> str:
            nonlocal prompt, image_base64
            try:
                llm: Llama = get_llm()
                
                if image_base64 and getattr(llm, "chat_handler", None) is not None:
                    print(f"[Model] Running vision query with image of size {len(image_base64)} characters", flush=True)
                    if not image_base64.startswith("data:image"):
                        image_base64 = f"data:image/jpeg;base64,{image_base64}"
                    
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
                    
                    # Ensure dynamic folders exist
                    global_cache_dir = Path("global_cache")
                    global_cache_dir.mkdir(exist_ok=True)
                    
                    # 1. Load Client Global Cache first (pre-compiled prefix)
                    global_cache_file = global_cache_dir / f"{client_id}.bin" if client_id else None
                    loaded_global = False
                    
                    if global_cache_file and global_cache_file.exists():
                        try:
                            print(f"[Model] Restoring client global cache: {global_cache_file.name}", flush=True)
                            with open(global_cache_file, "rb") as f:
                                global_state = pickle.load(f)
                            llm.load_state(global_state)
                            loaded_global = True
                        except Exception as e:
                            print(f"[Model] Warning: Failed to load global cache: {e}", flush=True)
                            llm.reset()
                    else:
                        llm.reset()
                        print("[Model] No client global cache found, running from scratch.", flush=True)
                    
                    # 2. Load User Convo History cache on top of the global prefix
                    convo_file = STATES_DIR / f"{phone_number}.bin" if phone_number else None
                    if convo_file and convo_file.exists() and loaded_global:
                        try:
                            print(f"[Model] Stapling conversation history: {convo_file.name}", flush=True)
                            with open(convo_file, "rb") as f:
                                convo_state = pickle.load(f)
                            llm.load_state(convo_state)
                        except Exception as e:
                            print(f"[Model] Warning: Failed to restore conversation history: {e}", flush=True)
                    
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
                    
                    # 3. Save updated conversation state (containing new query + reply)
                    if phone_number:
                        try:
                            state_obj = llm.save_state()
                            full_evaluated_text = formatted_prompt + text_result
                            full_tokens = llm.tokenize(full_evaluated_text.encode("utf-8"))
                            
                            t = threading.Thread(
                                target=save_state_bg,
                                args=(convo_file, state_obj, full_tokens),
                                daemon=True
                            )
                            t.start()
                        except Exception as save_err:
                            print(f"[Model] Warning: Failed to save updated state: {save_err}", flush=True)
                    
                return text_result
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"Exception raised while running llama-cpp: {e}"

        return await asyncio.to_thread(evaluate_query)