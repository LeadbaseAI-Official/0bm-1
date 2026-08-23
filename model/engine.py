from pathlib import Path
from typing import Optional
from llama_cpp import Llama, GGML_TYPE_Q8_0 # type: ignore
import threading
import asyncio

PARALLEL_SLOTS = 3
_concurrency_semaphore: Optional[asyncio.Semaphore] = None
_eval_lock = threading.Lock()
_llm_instance: Optional[Llama] = None

def log_message(tag: str, msg: str) -> None:
    from datetime import datetime as dt, timezone, timedelta
    ist_now = dt.now(timezone.utc) + timedelta(hours=5, minutes=30)
    stamp_str = ist_now.strftime("%H:%M | %d")
    print(f"[{stamp_str}] [{tag.upper()}] : \"{msg}\"", flush=True)


def find_gguf_file() -> Path:
    for path in Path(".").glob("*.gguf"):
        if "mmproj" not in path.name:
            return path
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
