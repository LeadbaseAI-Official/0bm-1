import threading
import queue
import pickle
from pathlib import Path
from typing import Dict, Any, List
from model.engine import get_llm, _eval_lock, log_message
from model.cache_manager import _ram_states_cache, STATES_DIR, save_state_bg

_rebuild_queue: queue.Queue = queue.Queue()
_is_rebuilding: bool = False
_rebuild_lock = threading.Lock()

def is_model_rebuilding() -> bool:
    """Returns True if global KV cache rebuild is currently in progress."""
    with _rebuild_lock:
        return _is_rebuilding

def _process_rebuild_queue() -> None:
    """Background worker thread processing global KV cache rebuild tasks."""
    global _is_rebuilding
    while True:
        task = _rebuild_queue.get()
        if not task:
            continue
        global_state_obj = task.get("global_state")
        global_tokens = task.get("global_tokens", [])
        
        log_message("rebuild", "Starting background KV cache rebuild for customer states...")
        llm = get_llm()
        
        for convo_file in STATES_DIR.glob("*.bin"):
            customer_key = convo_file.stem
            try:
                with open(convo_file, "rb") as f:
                    customer_obj = pickle.load(f)
                
                history = customer_obj.get("history", [])
                if not history:
                    continue
                
                # Re-tokenize raw transcript text on top of new global seed using Qwen ChatML
                transcript_parts = []
                for msg in history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        transcript_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
                    else:
                        transcript_parts.append(f"{content}<|im_end|>\n")
                        
                raw_transcript = "".join(transcript_parts)
                transcript_tokens = llm.tokenize(raw_transcript.encode("utf-8"))
                full_rebuilt_tokens = global_tokens + transcript_tokens
                
                with _eval_lock:
                    llm.load_state(global_state_obj)
                    llm.eval(transcript_tokens)
                    new_state = llm.save_state()
                    
                customer_obj["state"] = new_state
                customer_obj["tokens"] = full_rebuilt_tokens
                
                if customer_key in _ram_states_cache:
                    _ram_states_cache[customer_key] = customer_obj
                    
                save_state_bg(convo_file, customer_obj)
                log_message("rebuild", f"Rebuilt KV cache for customer={customer_key}.")
            except Exception as ex:
                log_message("rebuild", f"Error rebuilding KV cache for customer={customer_key}: {ex}")
                
        with _rebuild_lock:
            _is_rebuilding = False
        log_message("rebuild", "Completed global KV cache rebuild.")
        _rebuild_queue.task_done()

threading.Thread(target=_process_rebuild_queue, daemon=True).start()

def queue_global_rebuild(global_state: Any, global_tokens: List[int]) -> None:
    """Queues a background job to rebuild all active customer state KV caches against new global seed."""
    global _is_rebuilding
    with _rebuild_lock:
        _is_rebuilding = True
    _rebuild_queue.put({
        "global_state": global_state,
        "global_tokens": global_tokens
    })

