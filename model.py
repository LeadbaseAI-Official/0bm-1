from pathlib import Path
from typing import Optional, Dict, Any
from llama_cpp import Llama, GGML_TYPE_Q8_0 # type: ignore
from chat_template import format_chat_prompt # type: ignore
import pickle
import threading
import os
import asyncio
import datetime
import requests

# Create states directory
STATES_DIR = Path("states")
STATES_DIR.mkdir(parents=True, exist_ok=True)

_llm_lock = asyncio.Lock()
_llm_instance: Optional[Llama] = None
_states: Dict[str, Any] = {}

MODEL_CODE = "0bm"

# Standardized logging helper: [HH:MM:SS | DD] [tag] : msg
def log_message(tag: str, msg: str) -> None:
    now = datetime.datetime.now()
    now_str = now.strftime("%H:%M:%S")
    day_str = now.strftime("%d")
    print(f"[{now_str} | {day_str}] [{tag}] : {msg}", flush=True)

def find_gguf_file() -> Path:
    # Check current directory
    for path in Path(".").glob("*.gguf"):
        if "mmproj" not in path.name:
            return path
    # Check model/ directory
    model_dir: Path = Path("model")
    if model_dir.exists():
        for path in model_dir.glob("*.gguf"):
            if "mmproj" not in path.name:
                return path
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
                from llama_cpp.llama_chat_format import LlavaChatHandler # type: ignore
                log_message("system", f"Found vision projector file: {mmproj_path}")
                chat_handler = LlavaChatHandler(clip_model_path=str(mmproj_path))
            except Exception as e:
                log_message("system", f"Warning: Failed to load LlavaChatHandler: {e}")
        
        # Optimize context and quantization specs
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
        log_message("system", f"Background state saved to {state_file.name}")
    except Exception as e:
        log_message("system", f"Background state save warning: {e}")

async def run_model_query(prompt: str, client_id: Optional[str] = None, phone_number: Optional[str] = None, image_base64: Optional[str] = None) -> str:
    async with _llm_lock:
        def evaluate_query() -> str:
            nonlocal prompt, image_base64
            try:
                llm: Llama = get_llm()
                
                # Vision mode handling
                if image_base64 and getattr(llm, "chat_handler", None) is not None:
                    log_message("system", f"Running vision query with image of size {len(image_base64)} characters")
                    if not image_base64.startswith("data:image"):
                        image_base64 = f"data:image/jpeg;base64,{image_base64}"
                    
                    logit_bias = {}
                    try:
                        # Ban thought and thinking tokens
                        thought_token_id = llm.tokenize(b"<|channel>thought")[-1]
                        logit_bias[thought_token_id] = -100.0
                        think_id = llm.tokenize(b"<think>")[-1]
                        end_think_id = llm.tokenize(b"</think>")[-1]
                        logit_bias[think_id] = -100.0
                        logit_bias[end_think_id] = -100.0
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
                    for chunk in response_generator:
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            token_text = delta["content"]
                            text_chunks.append(token_text)
                    text_result = "".join(text_chunks)
                    log_message("response", text_result)
                else:
                    if image_base64:
                        log_message("system", f"Text fallback mode: Received image of size {len(image_base64)} characters")
                        prompt = f"[User uploaded an image. Base64 length: {len(image_base64)}]\n{prompt}"
                    
                    # Formulate query suffix in ChatML format
                    query_suffix: str = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    new_tokens = llm.tokenize(query_suffix.encode("utf-8"))
                    
                    # Ensure dynamic folders exist
                    global_cache_dir = Path("global_cache")
                    global_cache_dir.mkdir(exist_ok=True)
                    
                    # 1. Load Client Global Cache first (pre-compiled prefix)
                    global_cache_file = global_cache_dir / f"{client_id}.bin" if client_id else None
                    loaded_global = False
                    prefix_tokens = []
                    
                    if global_cache_file and global_cache_file.exists():
                        try:
                            log_message("system", f"Restoring client global cache: {global_cache_file.name}")
                            with open(global_cache_file, "rb") as f:
                                payload_obj = pickle.load(f)
                            
                            if isinstance(payload_obj, dict) and "state" in payload_obj:
                                llm.load_state(payload_obj["state"])
                                prefix_tokens = payload_obj.get("tokens", [])
                            else:
                                llm.load_state(payload_obj)
                                prefix_tokens = []
                                
                            loaded_global = True
                        except Exception as e:
                            log_message("system", f"Warning: Failed to load global cache: {e}")
                            llm.reset()
                    else:
                        llm.reset()
                        log_message("system", "No client global cache found, running from scratch.")
                    
                    # 2. Load User Convo History cache on top of the global prefix
                    convo_file = STATES_DIR / f"{phone_number}.bin" if phone_number else None
                    convo_tokens = []
                    
                    # On-demand state restoration if missing locally
                    if convo_file and not convo_file.exists() and phone_number and loaded_global:
                        org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")
                        try:
                            # Resolve active redis-worker URL
                            res_dns = requests.get(f"https://raw.githubusercontent.com/{org}/dns/main/config.json", timeout=5)
                            if res_dns.status_code == 200:
                                redis_url = res_dns.json().get("redis-worker", {}).get("active")
                                if redis_url:
                                    log_message("system", f"Local convo cache missing. Querying active Redis for state:{phone_number}...")
                                    res_redis = requests.get(f"{redis_url.rstrip('/')}/get?key=state:{phone_number}", timeout=10)
                                    if res_redis.status_code == 200:
                                        payload = res_redis.json().get("value", "")
                                        if payload:
                                            import gzip
                                            compressed_bytes = base64.b64decode(payload)
                                            decompressed = gzip.decompress(compressed_bytes)
                                            with open(convo_file, "wb") as f:
                                                f.write(decompressed)
                                            log_message("system", f"Successfully hydrated convo cache for {phone_number} from Redis.")
                        except Exception as redis_err:
                            log_message("system", f"Warning: Failed to fetch state for {phone_number} from Redis: {redis_err}")

                    if convo_file and convo_file.exists() and loaded_global:
                        try:
                            log_message("system", f"Stapling conversation history: {convo_file.name}")
                            with open(convo_file, "rb") as f:
                                convo_data = pickle.load(f)
                                
                            if isinstance(convo_data, dict) and "state" in convo_data:
                                llm.load_state(convo_data["state"])
                                convo_tokens = convo_data.get("tokens", [])
                            else:
                                llm.load_state(convo_data)
                                convo_tokens = []
                        except Exception as e:
                            log_message("system", f"Warning: Failed to restore conversation history: {e}")
                    
                    # Target token list evaluated so far
                    evaluated_tokens = convo_tokens if len(convo_tokens) > 0 else prefix_tokens
                    
                    # Append new prompt query suffix tokens
                    full_token_sequence = evaluated_tokens + new_tokens
                    
                    # Set current evaluation index inside the context cache
                    if len(evaluated_tokens) > 0:
                        llm.n_tokens = len(evaluated_tokens)
                        log_message("system", f"Recycling KV cache: Preserving {len(evaluated_tokens)} prefix tokens. Appending {len(new_tokens)} suffix tokens.")
                    else:
                        llm.reset()
                        log_message("system", f"Fresh context run: Evaluating all {len(full_token_sequence)} tokens.")

                    # Apply logit_bias to ban <|channel>thought and <think>/</think> Qwen reasoning tokens
                    logit_bias = {}
                    try:
                        thought_token_id = llm.tokenize(b"<|channel>thought")[-1]
                        logit_bias[thought_token_id] = -100.0
                        
                        think_id = llm.tokenize(b"<think>")[-1]
                        end_think_id = llm.tokenize(b"</think>")[-1]
                        logit_bias[think_id] = -100.0
                        logit_bias[end_think_id] = -100.0
                    except Exception:
                        pass

                    # Stream completion utilizing token array directly to preserve cache mapping
                    completion_generator = llm.create_completion(
                        prompt=full_token_sequence,
                        max_tokens=512,
                        stream=True,
                        temp=0.7,
                        top_k=40,
                        top_p=0.9,
                        logit_bias=logit_bias
                    )
                    
                    text_result_chunks = []
                    for chunk in completion_generator:
                        token_text = chunk["choices"][0]["text"]
                        # Filter out reasoning tokens if they bypass logit_bias
                        if "<think>" in token_text:
                            continue
                        text_result_chunks.append(token_text)
                        
                    text_result = "".join(text_result_chunks)
                    log_message("response", text_result)
                    
                    # 3. Save updated conversation state
                    if phone_number:
                        try:
                            state_obj = llm.save_state()
                            full_tokens = full_token_sequence + llm.tokenize(text_result.encode("utf-8"))
                            
                            t = threading.Thread(
                                target=save_state_bg,
                                args=(convo_file, state_obj, full_tokens),
                                daemon=True
                            )
                            t.start()
                        except Exception as save_err:
                            log_message("system", f"Warning: Failed to save updated state: {save_err}")
                    
                return text_result
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"Exception raised while running llama-cpp: {e}"

        return await asyncio.to_thread(evaluate_query)