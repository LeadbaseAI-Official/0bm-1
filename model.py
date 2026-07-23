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
_active_phone_number: Optional[str] = None

MODEL_CODE = "0bm"
MAX_HISTORY = 200

# Standardized logging helper: [HH:MM:SS | DD] [tag] : msg
def log_message(tag: str, msg: str) -> None:
    from datetime import datetime as dt, timezone, timedelta
    ist_now = dt.now(timezone.utc) + timedelta(hours=5, minutes=30)
    now_str = ist_now.strftime("%H:%M:%S")
    day_str = ist_now.strftime("%d")
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

def save_state_bg(state_file: Path, customer_obj: dict) -> None:
    try:
        tmp_file = state_file.with_suffix(f".{threading.get_ident()}.tmp")
        with open(tmp_file, "wb") as sf:
            pickle.dump(customer_obj, sf)
        os.replace(tmp_file, state_file)
        log_message("system", f"Background state saved to {state_file.name}")
    except Exception as e:
        log_message("system", f"Background state save warning: {e}")

async def run_model_query(prompt: str, client_id: Optional[str] = None, phone_number: Optional[str] = None, image_base64: Optional[str] = None) -> str:
    import base64
    async with _llm_lock:
        def evaluate_query() -> str:
            nonlocal prompt, image_base64
            global _active_phone_number
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
                    return text_result
                else:
                    if image_base64:
                        log_message("system", f"Text fallback mode: Received image of size {len(image_base64)} characters")
                        prompt = f"[User uploaded an image. Base64 length: {len(image_base64)}]\n{prompt}"
                    
                    # Ensure dynamic folders exist
                    global_cache_dir = Path("global_cache")
                    global_cache_dir.mkdir(exist_ok=True)
                    
                    new_turn_text = f"\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    new_turn_tokens = llm.tokenize(new_turn_text.encode("utf-8"))

                    convo_file = STATES_DIR / f"{phone_number}_phone.bin" if phone_number else None
                    convo_tokens = []
                    history = []
                    msg_count = 0
                    prompt_to_evaluate = new_turn_tokens

                    # Active RAM shortcut: If same phone number is already active in RAM, skip disk reload completely!
                    if phone_number and _active_phone_number == phone_number and llm.n_tokens > 0:
                        log_message("system", f"Phone {phone_number} active in RAM ({llm.n_tokens} tokens). Appending {len(new_turn_tokens)} suffix tokens.")
                        if convo_file and convo_file.exists():
                            try:
                                with open(convo_file, "rb") as f:
                                    customer_obj = pickle.load(f)
                                if isinstance(customer_obj, dict):
                                    history = customer_obj.get("history", [])
                                    msg_count = customer_obj.get("msg_count", 0)
                                    convo_tokens = customer_obj.get("tokens", [])
                            except Exception:
                                pass
                    else:
                        # 1. Load Client Global Cache first (pre-compiled prefix)
                        global_cache_file = global_cache_dir / f"{client_id}_global.bin" if client_id else None
                        if global_cache_file and not global_cache_file.exists():
                            global_cache_file = global_cache_dir / f"{client_id}.bin"
                        
                        loaded_global = False
                        prefix_tokens = []
                        global_cache_state = None
                        
                        if global_cache_file and global_cache_file.exists():
                            try:
                                log_message("system", f"Restoring client global cache: {global_cache_file.name}")
                                with open(global_cache_file, "rb") as f:
                                    payload_obj = pickle.load(f)
                                
                                if isinstance(payload_obj, dict) and "state" in payload_obj:
                                    global_cache_state = payload_obj["state"]
                                    prefix_tokens = payload_obj.get("tokens", [])
                                else:
                                    global_cache_state = payload_obj
                                    prefix_tokens = []
                                    
                                loaded_global = True
                            except Exception as e:
                                log_message("system", f"Warning: Failed to load global cache: {e}")
                        
                        # 2. Load User Convo History cache from phonenumber_phone.bin
                        loaded_convo = False
                        if convo_file and not convo_file.exists() and phone_number and loaded_global:
                            org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")
                            try:
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

                        if convo_file and convo_file.exists():
                            try:
                                log_message("system", f"Loading conversation history object: {convo_file.name}")
                                with open(convo_file, "rb") as f:
                                    customer_obj = pickle.load(f)
                                    
                                if isinstance(customer_obj, dict) and "state" in customer_obj:
                                    llm.load_state(customer_obj["state"])
                                    convo_tokens = customer_obj.get("tokens", [])
                                    history = customer_obj.get("history", [])
                                    msg_count = customer_obj.get("msg_count", 0)
                                    loaded_convo = True
                                else:
                                    llm.load_state(customer_obj)
                                    convo_tokens = []
                                    history = []
                                    msg_count = 0
                                    loaded_convo = True
                            except Exception as e:
                                log_message("system", f"Warning: Failed to restore conversation history: {e}")
                        
                        if not loaded_convo:
                            if loaded_global and global_cache_state:
                                llm.load_state(global_cache_state)
                                convo_tokens = prefix_tokens
                                log_message("system", f"New user context: Initialized with {global_cache_file.name}")
                            else:
                                llm.reset()
                                convo_tokens = []
                                log_message("system", "No client global cache or conversation history, running from scratch.")
                                
                        _active_phone_number = phone_number
                        
                        # When loading state, pass only incremental turn tokens if KV state is loaded
                        if llm.n_tokens > 0:
                            prompt_to_evaluate = new_turn_tokens
                        else:
                            prompt_to_evaluate = convo_tokens + new_turn_tokens

                    # Apply logit_bias to ban thought tokens and suppress <stop> token
                    logit_bias = {}
                    try:
                        thought_token_id = llm.tokenize(b"<|channel>thought")[-1]
                        logit_bias[thought_token_id] = -100.0
                        
                        think_id = llm.tokenize(b"<think>")[-1]
                        end_think_id = llm.tokenize(b"</think>")[-1]
                        logit_bias[think_id] = -100.0
                        logit_bias[end_think_id] = -100.0

                        # Suppress <stop> token — we only use <abandon> now
                        stop_token_id = llm.tokenize(b"<stop>")[-1]
                        logit_bias[stop_token_id] = -100.0
                    except Exception:
                        pass

                    completion_generator = llm.create_completion(
                        prompt=prompt_to_evaluate,
                        max_tokens=512,
                        stream=True,
                        temperature=0.7,
                        top_k=40,
                        top_p=0.9,
                        logit_bias=logit_bias,
                        stop=["<|im_end|>", "<|im_start|>", "<|im_end|}", "<|im_start|}", "<|endoftext|>"]
                    )
                    
                    text_result_chunks = []
                    for chunk in completion_generator:
                        token_text = chunk["choices"][0]["text"]
                        text_result_chunks.append(token_text)
                        
                    raw_text = "".join(text_result_chunks)
                    import re
                    cleaned_text = re.sub(r'<think>[\s\S]*?</think>', '', raw_text)
                    
                    # Cut off text cleanly at any ChatML tag or variant (e.g. <|im_start|, <|im_end|, <|im_start|}, etc.)
                    cleaned_text = re.split(r'<\|im_(?:start|end)[\|>\}]?', cleaned_text)[0]
                    
                    # Extract <abandon> token before stripping it from visible reply
                    abandon_token: Optional[str] = None
                    abandon_match = re.search(r'<abandon>(.*?)</abandon>', cleaned_text, re.IGNORECASE | re.DOTALL)
                    if abandon_match:
                        abandon_token = abandon_match.group(1).strip()
                    # Strip both <abandon> and legacy <stop> tags from the visible reply
                    cleaned_text = re.sub(r'<abandon>[\s\S]*?</abandon>', '', cleaned_text, flags=re.IGNORECASE)
                    cleaned_text = re.sub(r'<stop>[\s\S]*?</stop>', '', cleaned_text, flags=re.IGNORECASE)
                    text_result = cleaned_text.strip()
                    
                    if abandon_token:
                        greetings = ["hi", "hello", "good morning", "good afternoon", "good evening", "hey"]
                        lower_reply = text_result.lower()
                        if any(g in lower_reply for g in greetings) and len(text_result) < 150:
                            log_message("system", f"Ignored false-positive abandon token '{abandon_token}' because response is a greeting.")
                            abandon_token = None
                            
                    log_message("response", f"{text_result}{' [ABANDON:' + abandon_token + ']' if abandon_token else ''}")
                    
                    # 4. Save updated conversation state
                    if phone_number:
                        try:
                            # Append user prompt and assistant response to text history
                            history.append({"role": "user", "content": prompt})
                            history.append({"role": "assistant", "content": text_result})
                            msg_count += 2
                            
                            state_obj = llm.save_state()
                            full_tokens = convo_tokens + new_turn_tokens + llm.tokenize(text_result.encode("utf-8")) + llm.tokenize(b"<|im_end|>\n")
                            
                            customer_obj = {
                                "phone_number": phone_number,
                                "state": state_obj,
                                "tokens": full_tokens,
                                "history": history,
                                "msg_count": msg_count
                            }
                            
                            t = threading.Thread(
                                target=save_state_bg,
                                args=(convo_file, customer_obj),
                                daemon=True
                            )
                            t.start()
                            
                            # Check if msg_count reaches MAX_HISTORY (200 messages) to exclude the number persistently
                            if msg_count >= MAX_HISTORY:
                                abandon_token = "MAX_LIMIT_REACHED"
                                log_message("system", f"Phone number {phone_number} reached MAX_HISTORY limit ({msg_count} msgs). Excluding in Redis...")
                                try:
                                    org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")
                                    pat: str = os.getenv("GITHUB_PAT", "")
                                    headers = {
                                        "User-Agent": "LeadBaseAI-Runner",
                                        "Cache-Control": "no-cache, no-store, must-revalidate",
                                        "Pragma": "no-cache"
                                    }
                                    if pat:
                                        headers["Authorization"] = f"token {pat}"
                                        
                                    config_data = None
                                    try:
                                        api_url = f"https://api.github.com/repos/{org}/dns/contents/config.json"
                                        res_api = requests.get(api_url, headers=headers, timeout=5)
                                        if res_api.status_code == 200:
                                            api_json = res_api.json()
                                            if "content" in api_json:
                                                import json
                                                decoded = base64.b64decode(api_json["content"]).decode("utf-8")
                                                config_data = json.loads(decoded)
                                    except Exception:
                                        pass
                                        
                                    if not config_data:
                                        import time
                                        timestamp = int(time.time())
                                        res_dns = requests.get(f"https://raw.githubusercontent.com/{org}/dns/main/config.json?t={timestamp}", headers=headers, timeout=5)
                                        if res_dns.status_code == 200:
                                            config_data = res_dns.json()

                                    if config_data:
                                        redis_url = config_data.get("redis-worker", {}).get("active")
                                        if redis_url:
                                            requests.post(
                                                f"{redis_url.rstrip('/')}/add",
                                                json={"key": f"excluded:{phone_number}", "value": "true"},
                                                timeout=5
                                            )
                                            log_message("system", f"Successfully marked excluded:{phone_number} in Redis.")
                                except Exception as ex_err:
                                    log_message("system", f"Warning: Failed to publish excluded status to Redis: {ex_err}")
                        except Exception as save_err:
                            log_message("system", f"Warning: Failed to save updated state: {save_err}")
                    
                    return {"response": text_result, "abandon_token": abandon_token}
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"Exception raised while running llama-cpp: {e}"

        return await asyncio.to_thread(evaluate_query)