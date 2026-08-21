# Thin backward-compatibility wrapper importing modular package components
from model.engine import get_llm, log_message, PARALLEL_SLOTS, find_gguf_file, find_mmproj_file
from model.cache_manager import save_state_bg, push_state_to_redis, enqueue_to_ssd_retry, _ram_states_cache, RAM_CACHE_CAPACITY, STATES_DIR, GLOBAL_CACHE_DIR
from model.query_processor import run_model_query, MAX_HISTORY
from model.rebuilder import queue_global_rebuild, is_client_rebuilding
from model.lifecycle import run_3_45_lifecycle_timer

MODEL_CODE = "0bm"

__all__ = [
    "run_model_query",
    "get_llm",
    "log_message",
    "save_state_bg",
    "push_state_to_redis",
    "enqueue_to_ssd_retry",
    "queue_global_rebuild",
    "run_3_45_lifecycle_timer",
    "MODEL_CODE"
]