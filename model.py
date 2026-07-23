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

from collections import OrderedDict

PARALLEL_SLOTS = 3
_concurrency_semaphore: Optional[asyncio.Semaphore] = None
_eval_lock = threading.Lock()
_llm_instance: Optional[Llama] = None

RAM_CACHE_CAPACITY = 5
_ram_states_cache: OrderedDict[str, dict] = OrderedDict()

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
        log_message("system", "Single model weight instance initialized successfully (Supports 3 Parallel Concurrency Slots).")
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
    global _concurrency_semaphore
    if _concurrency_semaphore is None:
        _concurrency_semaphore = asyncio.Semaphore(PARALLEL_SLOTS)

    async with _concurrency_semaphore:
        def evaluate_query() -> str:
            nonlocal prompt, image_base64
            global _ram_states_cache
            with _eval_lock:
                try:
                    llm: Llama = get_llm()
                    log_message("debug", f"═══════════════════════════════════════════════════════════")
                    log_message("debug", f"INCOMING REQUEST")
                    log_message("debug", f"  phone_number  = {phone_number}")
                    log_message("debug", f"  client_id     = {client_id}")
                    log_message("debug", f"  prompt        = {repr(prompt[:100])}{'...' if len(prompt) > 100 else ''}")
                    log_message("debug", f"  image_base64  = {'YES (' + str(len(image_base64)) + ' chars)' if image_base64 else 'None'}")
                    log_message("debug", f"  llm.n_tokens  = {llm.n_tokens} (before any load)")
                    log_message("debug", f"  RAM cache     = {list(_ram_states_cache.keys())} ({len(_ram_states_cache)}/{RAM_CACHE_CAPACITY})")
                    log_message("debug", f"═══════════════════════════════════════════════════════════")
                    
                    # Vision mode handling
                    if image_base64 and getattr(llm, "chat_handler", None) is not None:
                        log_message("system", f"Running vision query with image of size {len(image_base64)} characters")
                        if not image_base64.startswith("data:image"):
                            image_base64 = f"data:image/jpeg;base64,{image_base64}"

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

                    # ─── STEP 1: Tokenize new user turn ───
                    global_cache_dir = Path("global_cache")
                    global_cache_dir.mkdir(exist_ok=True)
                    
                    new_turn_text = f"\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    new_turn_tokens = llm.tokenize(new_turn_text.encode("utf-8"))
                    log_message("debug", f"STEP 1: Tokenized new turn")
                    log_message("debug", f"  new_turn_text (first 80 chars) = {repr(new_turn_text[:80])}")
                    log_message("debug", f"  new_turn_tokens count          = {len(new_turn_tokens)}")
                    log_message("debug", f"  new_turn_tokens[:5]            = {new_turn_tokens[:5]}")

                    convo_file = STATES_DIR / f"{phone_number}_phone.bin" if phone_number else None
                    convo_tokens = []
                    history = []
                    msg_count = 0
                    loaded_convo = False
                    load_source = "NONE"
                    loaded_n_tokens = 0  # Truth from KV cache after load_state

                    # ─── STEP 2: Try RAM LRU Cache (Level-1) ───
                    if phone_number and phone_number in _ram_states_cache:
                        ram_obj = _ram_states_cache[phone_number]
                        _ram_states_cache.move_to_end(phone_number)
                        llm.load_state(ram_obj["state"])
                        loaded_n_tokens = llm.n_tokens  # KV cache truth
                        history = ram_obj.get("history", [])
                        msg_count = ram_obj.get("msg_count", 0)
                        convo_tokens = ram_obj.get("tokens", [])
                        loaded_convo = True
                        load_source = "RAM_LRU"
                        log_message("debug", f"STEP 2: RAM LRU HIT ✓")
                        log_message("debug", f"  phone          = {phone_number}")
                        log_message("debug", f"  convo_tokens   = {len(convo_tokens)} tokens loaded from RAM")
                        log_message("debug", f"  loaded_n_tokens= {loaded_n_tokens} (KV cache truth)")
                        log_message("debug", f"  history turns  = {len(history)} messages")
                        log_message("debug", f"  msg_count      = {msg_count}")
                        log_message("debug", f"  SYNC CHECK     = {'✓ MATCH' if loaded_n_tokens == len(convo_tokens) else '✗ MISMATCH! KV=' + str(loaded_n_tokens) + ' tokens=' + str(len(convo_tokens))}")
                        log_message("debug", f"  tokens[:5]     = {convo_tokens[:5] if convo_tokens else '[]'}")
                        log_message("debug", f"  tokens[-5:]    = {convo_tokens[-5:] if convo_tokens else '[]'}")
                    else:
                        log_message("debug", f"STEP 2: RAM LRU MISS ✗ for {phone_number}")
                        
                        # ─── STEP 3: Load Global Prefix Cache ───
                        global_cache_file = global_cache_dir / f"{client_id}_global.bin" if client_id else None
                        if global_cache_file and not global_cache_file.exists():
                            global_cache_file = global_cache_dir / f"{client_id}.bin"
                        
                        loaded_global = False
                        prefix_tokens = []
                        global_cache_state = None
                        
                        if global_cache_file and global_cache_file.exists():
                            try:
                                with open(global_cache_file, "rb") as f:
                                    payload_obj = pickle.load(f)
                                
                                if isinstance(payload_obj, dict) and "state" in payload_obj:
                                    global_cache_state = payload_obj["state"]
                                    prefix_tokens = payload_obj.get("tokens", [])
                                else:
                                    global_cache_state = payload_obj
                                    prefix_tokens = []
                                    
                                loaded_global = True
                                log_message("debug", f"STEP 3: Global prefix cache LOADED ✓")
                                log_message("debug", f"  file           = {global_cache_file.name}")
                                log_message("debug", f"  prefix_tokens  = {len(prefix_tokens)} tokens")
                                log_message("debug", f"  prefix[:5]     = {prefix_tokens[:5] if prefix_tokens else '[]'}")
                            except Exception as e:
                                log_message("debug", f"STEP 3: Global prefix cache FAILED ✗: {e}")
                        else:
                            log_message("debug", f"STEP 3: Global prefix cache NOT FOUND ✗ (file={global_cache_file})")
                        
                        # ─── STEP 4: Try Redis hydration ───
                        if convo_file and not convo_file.exists() and phone_number and loaded_global:
                            org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")
                            try:
                                res_dns = requests.get(f"https://raw.githubusercontent.com/{org}/dns/main/config.json", timeout=5)
                                if res_dns.status_code == 200:
                                    redis_url = res_dns.json().get("redis-worker", {}).get("active")
                                    if redis_url:
                                        log_message("debug", f"STEP 4: Querying Redis for state:{phone_number}...")
                                        res_redis = requests.get(f"{redis_url.rstrip('/')}/get?key=state:{phone_number}", timeout=10)
                                        if res_redis.status_code == 200:
                                            payload = res_redis.json().get("value", "")
                                            if payload:
                                                import gzip
                                                compressed_bytes = base64.b64decode(payload)
                                                decompressed = gzip.decompress(compressed_bytes)
                                                with open(convo_file, "wb") as f:
                                                    f.write(decompressed)
                                                log_message("debug", f"STEP 4: Redis hydration SUCCESS ✓ ({len(decompressed)} bytes)")
                                            else:
                                                log_message("debug", f"STEP 4: Redis returned empty payload ✗")
                                        else:
                                            log_message("debug", f"STEP 4: Redis returned HTTP {res_redis.status_code} ✗")
                                    else:
                                        log_message("debug", f"STEP 4: No active redis-worker in DNS ✗")
                            except Exception as redis_err:
                                log_message("debug", f"STEP 4: Redis hydration FAILED ✗: {redis_err}")
                        else:
                            disk_exists = convo_file and convo_file.exists()
                            log_message("debug", f"STEP 4: Redis skipped (disk_exists={disk_exists}, loaded_global={loaded_global if 'loaded_global' in dir() else 'N/A'})")

                        # ─── STEP 5: Load Disk State (Level-2) ───
                        if convo_file and convo_file.exists():
                            try:
                                with open(convo_file, "rb") as f:
                                    customer_obj = pickle.load(f)
                                    
                                if isinstance(customer_obj, dict) and "state" in customer_obj:
                                    llm.load_state(customer_obj["state"])
                                    loaded_n_tokens = llm.n_tokens  # KV cache truth
                                    convo_tokens = customer_obj.get("tokens", [])
                                    history = customer_obj.get("history", [])
                                    msg_count = customer_obj.get("msg_count", 0)
                                    loaded_convo = True
                                    load_source = "DISK"
                                else:
                                    llm.load_state(customer_obj)
                                    loaded_n_tokens = llm.n_tokens  # KV cache truth
                                    convo_tokens = []
                                    history = []
                                    msg_count = 0
                                    loaded_convo = True
                                    load_source = "DISK_LEGACY"
                                log_message("debug", f"STEP 5: Disk state LOADED ✓ (source={load_source})")
                                log_message("debug", f"  file           = {convo_file.name}")
                                log_message("debug", f"  convo_tokens   = {len(convo_tokens)} tokens")
                                log_message("debug", f"  loaded_n_tokens= {loaded_n_tokens} (KV cache truth)")
                                log_message("debug", f"  history turns  = {len(history)} messages")
                                log_message("debug", f"  msg_count      = {msg_count}")
                                log_message("debug", f"  SYNC CHECK     = {'✓ MATCH' if loaded_n_tokens == len(convo_tokens) else '✗ MISMATCH! KV=' + str(loaded_n_tokens) + ' tokens=' + str(len(convo_tokens))}")
                                log_message("debug", f"  tokens[:5]     = {convo_tokens[:5] if convo_tokens else '[]'}")
                                log_message("debug", f"  tokens[-5:]    = {convo_tokens[-5:] if convo_tokens else '[]'}")
                            except Exception as e:
                                log_message("debug", f"STEP 5: Disk state FAILED ✗: {e}")
                        
                        if not loaded_convo:
                            if loaded_global and global_cache_state:
                                llm.load_state(global_cache_state)
                                loaded_n_tokens = llm.n_tokens  # KV cache truth
                                convo_tokens = prefix_tokens
                                load_source = "GLOBAL_PREFIX"
                                log_message("debug", f"STEP 5: Initialized from global prefix (source={load_source})")
                                log_message("debug", f"  convo_tokens   = {len(convo_tokens)} (= prefix_tokens)")
                                log_message("debug", f"  loaded_n_tokens= {loaded_n_tokens} (KV cache truth)")
                            else:
                                llm.reset()
                                convo_tokens = []
                                load_source = "SCRATCH"
                                log_message("debug", f"STEP 5: Running from SCRATCH ✗ (no cache found)")

                    # ─── STEP 6: Build all_tokens & set n_tokens for prefix matching ───
                    evaluated_tokens = convo_tokens if (loaded_convo or (phone_number and phone_number in _ram_states_cache)) else (prefix_tokens if 'prefix_tokens' in dir() else [])
                    all_tokens = evaluated_tokens + new_turn_tokens
                    
                    # Exact timeline alignment logic:
                    # Compare evaluated_tokens array against actual internal input_ids in the model KV cache.
                    # Find exact longest matching prefix index to prevent full re-evaluation on partial KV cache removal warnings.
                    n_tokens_before_set = llm.n_tokens
                    target_match_len = loaded_n_tokens if loaded_n_tokens > 0 else len(evaluated_tokens)
                    
                    # Verify token-by-token alignment if input_ids is available
                    try:
                        kv_input_ids = llm.input_ids.tolist()[:target_match_len]
                        if len(kv_input_ids) > 0 and len(evaluated_tokens) >= len(kv_input_ids):
                            # Find longest exact matching prefix boundary
                            match_idx = 0
                            for idx in range(min(len(kv_input_ids), len(evaluated_tokens))):
                                if kv_input_ids[idx] == evaluated_tokens[idx]:
                                    match_idx += 1
                                else:
                                    break
                            if match_idx < target_match_len:
                                log_message("debug", f"STEP 6: ⚠ TIMELINE MISMATCH AT POS {match_idx}/{target_match_len}!")
                                log_message("debug", f"  Rewinding timeline to matching prefix index {match_idx}")
                                target_match_len = match_idx
                                evaluated_tokens = evaluated_tokens[:target_match_len]
                                all_tokens = evaluated_tokens + new_turn_tokens
                    except Exception:
                        pass

                    llm.n_tokens = target_match_len
                    
                    log_message("debug", f"STEP 6: Token alignment for prefix matching")
                    log_message("debug", f"  load_source         = {load_source}")
                    log_message("debug", f"  loaded_n_tokens     = {loaded_n_tokens} (KV cache truth)")
                    log_message("debug", f"  target_match_len    = {target_match_len}")
                    log_message("debug", f"  evaluated_tokens    = {len(evaluated_tokens)}")
                    log_message("debug", f"  new_turn_tokens     = {len(new_turn_tokens)}")
                    log_message("debug", f"  all_tokens          = {len(all_tokens)} (= evaluated + new_turn)")
                    log_message("debug", f"  llm.n_tokens BEFORE = {n_tokens_before_set}")
                    log_message("debug", f"  llm.n_tokens SET TO = {llm.n_tokens}")
                    log_message("debug", f"  MATCH EXPECTED      = {llm.n_tokens == len(evaluated_tokens)}")
                    if len(evaluated_tokens) > 0:
                        log_message("debug", f"  eval_tokens[:5]     = {evaluated_tokens[:5]}")
                        log_message("debug", f"  eval_tokens[-5:]    = {evaluated_tokens[-5:]}")
                        log_message("debug", f"  all_tokens[{len(evaluated_tokens)}:{len(evaluated_tokens)+5}] = {all_tokens[len(evaluated_tokens):len(evaluated_tokens)+5]}")

                    # ─── STEP 7: Apply logit_bias ───
                    logit_bias = {}
                    try:
                        thought_token_id = llm.tokenize(b"<|channel>thought")[-1]
                        logit_bias[thought_token_id] = -100.0
                        think_id = llm.tokenize(b"<think>")[-1]
                        end_think_id = llm.tokenize(b"</think>")[-1]
                        logit_bias[think_id] = -100.0
                        logit_bias[end_think_id] = -100.0
                        stop_token_id = llm.tokenize(b"<stop>")[-1]
                        logit_bias[stop_token_id] = -100.0
                    except Exception:
                        pass
                    log_message("debug", f"STEP 7: logit_bias = {len(logit_bias)} token(s) banned")

                    # ─── STEP 8 & 9: Evaluation & Streaming with Timeline Fallback ───
                    log_message("debug", f"STEP 8: Calling create_completion(prompt={len(all_tokens)} tokens, llm.n_tokens={llm.n_tokens})")
                    fallback_used = False
                    text_result_chunks = []
                    gen_token_count = 0
                    
                    try:
                        completion_generator = llm.create_completion(
                            prompt=all_tokens,
                            max_tokens=512,
                            stream=True,
                            temperature=0.7,
                            top_k=40,
                            top_p=0.9,
                            logit_bias=logit_bias,
                            stop=["<|im_end|>", "<|im_start|>", "<|im_end|}", "<|im_start|}", "<|endoftext|>"]
                        )
                        for chunk in completion_generator:
                            token_text = chunk["choices"][0]["text"]
                            text_result_chunks.append(token_text)
                            gen_token_count += 1
                        log_message("debug", f"STEP 8/9: Normal evaluation & generation completed OK ✓")
                    except Exception as eval_err:
                        fallback_used = True
                        text_result_chunks = []
                        gen_token_count = 0
                        log_message("debug", f"STEP 8 FALLBACK TRIGGERED ✗: {eval_err}")
                        
                        # Timeline Alignment Fallback:
                        # 1. Calculate exact matching boundary
                        match_len = min(llm.n_tokens, len(evaluated_tokens))
                        
                        # 2. Rewind evaluation pointer to matching index
                        llm.n_tokens = match_len
                        aligned_prompt = all_tokens[match_len:] if match_len > 0 else all_tokens
                        log_message("debug", f"STEP 8 FALLBACK: Timeline aligned to pos {match_len}")
                        log_message("debug", f"  llm.n_tokens rewound to {llm.n_tokens}")
                        log_message("debug", f"  aligned_prompt suffix = {len(aligned_prompt)} tokens")
                        log_message("debug", f"  aligned[:5]            = {aligned_prompt[:5] if len(aligned_prompt) > 0 else '[]'}")
                        
                        # 3. Retry evaluation from matching suffix point
                        completion_generator = llm.create_completion(
                            prompt=aligned_prompt,
                            max_tokens=512,
                            stream=True,
                            temperature=0.7,
                            top_k=40,
                            top_p=0.9,
                            logit_bias=logit_bias,
                            stop=["<|im_end|>", "<|im_start|>", "<|im_end|}", "<|im_start|}", "<|endoftext|>"]
                        )
                        for chunk in completion_generator:
                            token_text = chunk["choices"][0]["text"]
                            text_result_chunks.append(token_text)
                            gen_token_count += 1
                        log_message("debug", f"STEP 8/9 FALLBACK: Retry generation completed OK ✓")
                        
                    raw_text = "".join(text_result_chunks)
                    import re
                    cleaned_text = re.sub(r'<think>[\s\S]*?</think>', '', raw_text)
                    cleaned_text = re.split(r'<\|im_(?:start|end)[\|>\}]?', cleaned_text)[0]
                    
                    abandon_token: Optional[str] = None
                    abandon_match = re.search(r'<abandon>(.*?)</abandon>', cleaned_text, re.IGNORECASE | re.DOTALL)
                    if abandon_match:
                        abandon_token = abandon_match.group(1).strip()
                    cleaned_text = re.sub(r'<abandon>[\s\S]*?</abandon>', '', cleaned_text, flags=re.IGNORECASE)
                    cleaned_text = re.sub(r'<stop>[\s\S]*?</stop>', '', cleaned_text, flags=re.IGNORECASE)
                    text_result = cleaned_text.strip()
                    
                    if abandon_token:
                        greetings = ["hi", "hello", "good morning", "good afternoon", "good evening", "hey"]
                        lower_reply = text_result.lower()
                        if any(g in lower_reply for g in greetings) and len(text_result) < 150:
                            log_message("debug", f"STEP 9: Ignored false-positive abandon '{abandon_token}' (greeting)")
                            abandon_token = None
                    
                    log_message("debug", f"STEP 9: Generation complete")
                    log_message("debug", f"  gen_token_count  = {gen_token_count}")
                    log_message("debug", f"  raw_text length  = {len(raw_text)} chars")
                    log_message("debug", f"  text_result len  = {len(text_result)} chars")
                    log_message("debug", f"  abandon_token    = {abandon_token}")
                    log_message("debug", f"  fallback_used    = {fallback_used}")
                    log_message("debug", f"  llm.n_tokens     = {llm.n_tokens} (after generation)")
                    log_message("response", f"{text_result}{' [ABANDON:' + abandon_token + ']' if abandon_token else ''}")
                    
                    # ─── STEP 10: Save state ───
                    if phone_number:
                        try:
                            history.append({"role": "user", "content": prompt})
                            history.append({"role": "assistant", "content": text_result})
                            msg_count += 2
                            
                            state_obj = llm.save_state()
                            
                            # CRITICAL FIX: Get ACTUAL token IDs from llm internal state
                            # DO NOT re-tokenize raw_text — it produces DIFFERENT token IDs
                            # than what llama.cpp actually generated into the KV cache.
                            # e.g. model generates 71 tokens but tokenize(decode(71 tokens)) = 73 tokens
                            actual_n = llm.n_tokens
                            try:
                                full_tokens = llm.input_ids.tolist()
                                token_source = "input_ids"
                            except Exception:
                                try:
                                    full_tokens = list(llm._input_ids[:actual_n])
                                    token_source = "_input_ids"
                                except Exception:
                                    # Last resort fallback: re-tokenize (may cause mismatch)
                                    response_tokens = llm.tokenize(raw_text.encode("utf-8")) + llm.tokenize(b"<|im_end|>\n")
                                    full_tokens = all_tokens + response_tokens
                                    token_source = "re-tokenized (FALLBACK)"
                            
                            customer_obj = {
                                "phone_number": phone_number,
                                "state": state_obj,
                                "tokens": full_tokens,
                                "history": history,
                                "msg_count": msg_count
                            }
                            
                            log_message("debug", f"STEP 10: Saving state")
                            log_message("debug", f"  token_source     = {token_source}")
                            log_message("debug", f"  llm.n_tokens     = {actual_n} (KV cache truth)")
                            log_message("debug", f"  full_tokens      = {len(full_tokens)} (SAVED)")
                            log_message("debug", f"  SYNC CHECK       = {'✓ EXACT MATCH' if len(full_tokens) == actual_n else '✗ MISMATCH! saved=' + str(len(full_tokens)) + ' kv=' + str(actual_n)}")
                            log_message("debug", f"  full[:5]         = {full_tokens[:5]}")
                            log_message("debug", f"  full[-5:]        = {full_tokens[-5:]}")
                            log_message("debug", f"  state_obj size   = {len(state_obj) if hasattr(state_obj, '__len__') else 'N/A'}")
                            log_message("debug", f"  history turns    = {len(history)} messages")
                            log_message("debug", f"  msg_count        = {msg_count}")
                            
                            # Cache active session in Level-1 RAM LRU cache (Capacity: 5)
                            _ram_states_cache[phone_number] = customer_obj
                            _ram_states_cache.move_to_end(phone_number)
                            ram_evicted_phone = None
                            if len(_ram_states_cache) > RAM_CACHE_CAPACITY:
                                ram_evicted_phone, ram_evicted_obj = _ram_states_cache.popitem(last=False)
                                evicted_file = STATES_DIR / f"{ram_evicted_phone}_phone.bin"
                                t = threading.Thread(
                                    target=save_state_bg,
                                    args=(evicted_file, ram_evicted_obj),
                                    daemon=True
                                )
                                t.start()
                                log_message("debug", f"STEP 10: RAM cache capacity (5) reached. Evicted {ram_evicted_phone} to disk ({evicted_file.name})")
                            else:
                                log_message("debug", f"STEP 10: State updated in RAM LRU cache (0 disk I/O for active session).")
                            
                            log_message("debug", f"  RAM keys         = {list(_ram_states_cache.keys())}")
                            log_message("debug", f"  RAM size         = {len(_ram_states_cache)}/{RAM_CACHE_CAPACITY}")
                            log_message("debug", f"  RAM evicted      = {ram_evicted_phone or 'None'}")
                            
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
                            log_message("debug", f"STEP 10: SAVE FAILED ✗: {save_err}")
                    
                    log_message("debug", f"═══════════════════════════════════════════════════════════")
                    log_message("debug", f"REQUEST COMPLETE")
                    log_message("debug", f"  phone        = {phone_number}")
                    log_message("debug", f"  load_source  = {load_source}")
                    log_message("debug", f"  fallback     = {fallback_used}")
                    log_message("debug", f"  response len = {len(text_result)} chars")
                    log_message("debug", f"  abandon      = {abandon_token}")
                    log_message("debug", f"═══════════════════════════════════════════════════════════")
                    return {"response": text_result, "abandon_token": abandon_token}
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"Exception raised while running llama-cpp: {e}"
        return await asyncio.to_thread(evaluate_query)