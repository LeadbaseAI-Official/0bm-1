import os
import re
import pickle
import asyncio
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

from model.engine import (
    get_llm, log_message, PARALLEL_SLOTS, _eval_lock, _concurrency_semaphore
)
from model.cache_manager import (
    _ram_states_cache, RAM_CACHE_CAPACITY, STATES_DIR, GLOBAL_CACHE_DIR, save_state_bg
)
from model.rebuilder import is_model_rebuilding

MAX_HISTORY = 200

_jid_locks: Dict[str, asyncio.Lock] = {}
_jid_locks_guard = asyncio.Lock()

def ensure_input_ids_capacity(llm: Any) -> None:
    """
    Resizes llm.input_ids NumPy array after loading a compact state file
    so it has full context capacity for incoming conversation turns.
    """
    try:
        target_capacity: int = llm.n_ctx()
        current_len: int = len(llm.input_ids)
        if current_len < target_capacity:
            log_message("debug", f"Auto-expanding llm.input_ids from {current_len} to full model context capacity ({target_capacity})...")
            new_arr = np.zeros(target_capacity, dtype=np.int32)
            new_arr[:current_len] = llm.input_ids
            llm.input_ids = new_arr
    except Exception as err:
        log_message("debug", f"Warning: Failed to expand llm.input_ids capacity: {err}")


async def get_jid_lock(jid_key: str) -> asyncio.Lock:
    """Retrieves or creates a per-JID asyncio Lock to enforce sequential request processing per user."""
    async with _jid_locks_guard:
        if jid_key not in _jid_locks:
            _jid_locks[jid_key] = asyncio.Lock()
        return _jid_locks[jid_key]

async def run_model_query(
    prompt: str,
    phone_number: Optional[str] = None,
    clean_number: Optional[str] = None,
    jid: Optional[str] = None,
    image_base64: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates an incoming prompt against the Llama model using 3-slot concurrency
    and per-JID sequential locking for KV state consistency.
    """
    global _concurrency_semaphore
    if _concurrency_semaphore is None:
        _concurrency_semaphore = asyncio.Semaphore(PARALLEL_SLOTS)

    # Determine primary customer key (e.g. cleanNumber, phone_number, or JID)
    customer_key: str = clean_number or phone_number or jid or "default"
    jid_lock = await get_jid_lock(customer_key)

    # Enforce sequential execution per customer_key while allowing 3-slot slot parallelism
    async with jid_lock:
        async with _concurrency_semaphore:
            def evaluate_query() -> Dict[str, Any]:
                nonlocal prompt, image_base64
                global _ram_states_cache
                with _eval_lock:
                    try:
                        llm = get_llm()
                        log_message("debug", f"═══════════════════════════════════════════════════════════")
                        log_message("debug", f"INCOMING REQUEST: customer_key={customer_key}")
                        log_message("debug", f"  prompt        = {repr(prompt[:100])}{'...' if len(prompt) > 100 else ''}")
                        log_message("debug", f"  RAM cache     = {list(_ram_states_cache.keys())} ({len(_ram_states_cache)}/{RAM_CACHE_CAPACITY})")
                        log_message("debug", f"═══════════════════════════════════════════════════════════")

                        # Check if global prompt is actively undergoing KV cache rebuild
                        if is_model_rebuilding():
                            log_message("debug", "Global KV cache rebuild in progress. Temporarily holding query.")
                            return {"response": "System maintenance in progress. Please try again shortly.", "abandon_token": None}

                        # Vision mode handling
                        if image_base64 and getattr(llm, "chat_handler", None) is not None:
                            log_message("system", f"Running vision query with image size {len(image_base64)} chars")
                            if not image_base64.startswith("data:image"):
                                image_base64 = f"data:image/jpeg;base64,{image_base64}"
                            res_gen = llm.create_chat_completion(
                                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_base64}}]}],
                                max_tokens=512, stream=True
                            )
                            chunks = [c["choices"][0]["delta"]["content"] for c in res_gen if "content" in c["choices"][0]["delta"]]
                            text_res = "".join(chunks)
                            return {"response": text_res, "abandon_token": None}
                        elif image_base64:
                            prompt = f"[User uploaded an image. Base64 length: {len(image_base64)}]\n{prompt}"

                        # ─── STEP 1: Qwen ChatML Turn Tokenization ───
                        new_turn_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                        new_turn_tokens = llm.tokenize(new_turn_text.encode("utf-8"))

                        convo_file = STATES_DIR / f"{customer_key}.bin"
                        history: List[Dict[str, str]] = []
                        msg_count: int = 0
                        loaded_convo: bool = False

                        # ─── STEP 2: Level-1 RAM LRU Cache ───
                        if customer_key in _ram_states_cache:
                            ram_obj = _ram_states_cache[customer_key]
                            _ram_states_cache.move_to_end(customer_key)
                            llm.load_state(ram_obj["state"])
                            ensure_input_ids_capacity(llm)
                            history = ram_obj.get("history", [])
                            msg_count = ram_obj.get("msg_count", 0)
                            loaded_convo = True
                            log_message("debug", f"STEP 2: RAM LRU HIT ✓ (n_tokens={llm.n_tokens})")
                        else:
                            log_message("debug", f"STEP 2: RAM LRU MISS ✗ for {customer_key}")

                            # ─── STEP 3: Level-2 Local NVMe Disk Cache ───
                            if convo_file.exists():
                                try:
                                    with open(convo_file, "rb") as f:
                                        customer_obj = pickle.load(f)
                                    if isinstance(customer_obj, dict) and "state" in customer_obj:
                                        llm.load_state(customer_obj["state"])
                                        ensure_input_ids_capacity(llm)
                                        history = customer_obj.get("history", [])
                                        msg_count = customer_obj.get("msg_count", 0)
                                        loaded_convo = True
                                        log_message("debug", f"STEP 3: DISK HIT ✓ (n_tokens={llm.n_tokens})")
                                except Exception as e:
                                    log_message("debug", f"STEP 3: DISK FAILED ✗: {e}")

                            # ─── STEP 4: Global Seed Fallback ───
                            if not loaded_convo:
                                global_cache_file = GLOBAL_CACHE_DIR / "global.bin"
                                if not global_cache_file.exists():
                                    global_files = list(GLOBAL_CACHE_DIR.glob("*.bin"))
                                    if global_files:
                                        global_cache_file = global_files[0]

                                loaded_global = False
                                global_cache_state = None

                                if global_cache_file and global_cache_file.exists():
                                    try:
                                        with open(global_cache_file, "rb") as f:
                                            payload_obj = pickle.load(f)
                                        if isinstance(payload_obj, dict) and "state" in payload_obj:
                                            global_cache_state = payload_obj["state"]
                                        else:
                                            global_cache_state = payload_obj
                                        loaded_global = True
                                    except Exception as e:
                                        log_message("debug", f"STEP 4: Global seed load failed: {e}")

                                global_tokens: List[int] = []
                                if loaded_global and global_cache_state:
                                    llm.load_state(global_cache_state)
                                    ensure_input_ids_capacity(llm)
                                    log_message("debug", f"STEP 4: Initialized from GLOBAL_SEED (n_tokens={llm.n_tokens})")

                                    if isinstance(payload_obj, dict) and "tokens" in payload_obj:
                                        global_tokens = payload_obj["tokens"]
                                        log_message("debug", f"═════════════════ GLOBAL SEED TOKEN DUMP (Total: {len(global_tokens)}) ═════════════════")
                                        log_message("debug", f"Global Token IDs: {global_tokens}")
                                        try:
                                            decoded_global = [llm.detokenize([t]).decode("utf-8", errors="ignore") for t in global_tokens]
                                            log_message("debug", f"Global Decoded Tokens: {decoded_global}")
                                        except Exception as dt_err:
                                            log_message("debug", f"Global detokenize err: {dt_err}")
                                        log_message("debug", f"═══════════════════════════════════════════════════════════════════════════════")
                                else:
                                    log_message("debug", "STEP 4: UNCONFIGURED MODEL FALLBACK")
                                    return {
                                        "response": "Sorry, I cannot process this request since the model is not configured.",
                                        "abandon_token": None
                                    }

                        log_message("debug", f"═════════════════ NEW TURN TOKEN DUMP (Total: {len(new_turn_tokens)}) ═════════════════")
                        log_message("debug", f"New Turn Token IDs: {new_turn_tokens}")
                        try:
                            decoded_turn = [llm.detokenize([t]).decode("utf-8", errors="ignore") for t in new_turn_tokens]
                            log_message("debug", f"New Turn Decoded Tokens: {decoded_turn}")
                        except Exception as dt_err2:
                            log_message("debug", f"New Turn detokenize err: {dt_err2}")
                        log_message("debug", f"Current LLM State n_tokens before generate: {llm.n_tokens}")
                        log_message("debug", f"═══════════════════════════════════════════════════════════════════════════════")

                        # ─── STEP 5: Direct Incremental Evaluation & Token Sampling ───
                        # Suppress reasoning <think> tokens via LogitsProcessorList
                        from llama_cpp import LogitsProcessorList

                        think_toks: List[int] = []
                        try:
                            think_toks = llm.tokenize("<think>".encode("utf-8"), add_bos=False)
                            log_message("debug", f"Suppressed reasoning token IDs: {think_toks}")
                        except Exception as tb_err:
                            log_message("debug", f"Could not calculate think token IDs: {tb_err}")

                        def suppress_think_logits_processor(input_ids_arr: np.ndarray, logits_arr: np.ndarray) -> np.ndarray:
                            for t in think_toks:
                                if t < len(logits_arr):
                                    logits_arr[t] = -1000.0
                            return logits_arr

                        logits_processors = LogitsProcessorList([suppress_think_logits_processor]) if think_toks else None

                        # Stop tokens for Qwen ChatML
                        stop_token_ids = set()
                        for s_str in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
                            try:
                                s_toks = llm.tokenize(s_str.encode("utf-8"), add_bos=False, special=True)
                                if len(s_toks) == 1:
                                    stop_token_ids.update(s_toks)
                            except Exception:
                                pass
                        try:
                            eos_id = llm.token_eos()
                            if eos_id is not None:
                                stop_token_ids.add(eos_id)
                        except Exception:
                            pass
                        log_message("debug", f"Configured stop token IDs: {stop_token_ids}")


                        start_n: int = llm.n_tokens
                        log_message("debug", f"Evaluating {len(new_turn_tokens)} new turn tokens starting at n_tokens={start_n}...")
                        llm.eval(new_turn_tokens)
                        log_message("debug", f"✅ Evaluated new turn tokens. Updated n_tokens={llm.n_tokens}")

                        text_result_chunks: List[str] = []
                        max_new_tokens: int = 512

                        for step in range(max_new_tokens):
                            token_id = llm.sample(
                                temp=0.7,
                                top_k=40,
                                top_p=0.9,
                                logits_processor=logits_processors
                            )
                            if token_id in stop_token_ids:
                                log_message("debug", f"Reached stop token ID: {token_id} at step {step}")
                                break

                            piece_str = llm.detokenize([token_id]).decode("utf-8", errors="ignore")
                            text_result_chunks.append(piece_str)
                            llm.eval([token_id])


                        raw_text = "".join(text_result_chunks)
                        cleaned_text = re.split(r'<\|im_(?:start|end)[\|>\}]?', raw_text)[0]


                        abandon_token: Optional[str] = None
                        abandon_match = re.search(r'<abandon>(.*?)</abandon>', cleaned_text, re.IGNORECASE | re.DOTALL)
                        if abandon_match:
                            abandon_token = abandon_match.group(1).strip()
                        cleaned_text = re.sub(r'<abandon>[\s\S]*?</abandon>', '', cleaned_text, flags=re.IGNORECASE)
                        text_result = cleaned_text.strip()

                        log_message("response", f"{text_result}{' [ABANDON:' + abandon_token + ']' if abandon_token else ''}")

                        # ─── STEP 6: Save State to RAM LRU & Disk ───
                        try:
                            history.append({"role": "user", "content": prompt})
                            history.append({"role": "assistant", "content": text_result})
                            msg_count += 2
                            state_obj = llm.save_state()
                            actual_n = llm.n_tokens
                            full_tokens = llm.input_ids.tolist()[:actual_n]

                            customer_obj = {
                                "customer_key": customer_key,
                                "state": state_obj,
                                "tokens": full_tokens,
                                "history": history,
                                "msg_count": msg_count
                            }

                            _ram_states_cache[customer_key] = customer_obj
                            _ram_states_cache.move_to_end(customer_key)
                            
                            # Level-1 RAM Eviction -> Level-2 NVMe Disk
                            if len(_ram_states_cache) > RAM_CACHE_CAPACITY:
                                evicted_key, evicted_obj = _ram_states_cache.popitem(last=False)
                                evicted_file = STATES_DIR / f"{evicted_key}.bin"
                                save_state_bg(evicted_file, evicted_obj)

                            if msg_count >= MAX_HISTORY:
                                abandon_token = "MAX_LIMIT_REACHED"
                        except Exception as save_err:
                            log_message("debug", f"STEP 6: SAVE FAILED ✗: {save_err}")

                        return {"response": text_result, "abandon_token": abandon_token}
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        return {"response": f"Exception raised while running llama-cpp: {e}", "abandon_token": None}
            return await asyncio.to_thread(evaluate_query)


