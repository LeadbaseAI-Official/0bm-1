import os
import time
import pickle
import threading
import shutil
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Dict, Any, List
from model.engine import log_message

STATES_DIR = Path("states")
STATES_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_CACHE_DIR = Path("global_cache")
GLOBAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

RAM_CACHE_CAPACITY = 20
_ram_states_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

HF_REPO_ID = "anisoleai/client-states"

def save_state_bg(state_file: Path, customer_obj: Dict[str, Any]) -> None:
    """Asynchronously writes a customer KV state dictionary to disk."""
    try:
        tmp_file = state_file.with_suffix(f".{threading.get_ident()}.tmp")
        with open(tmp_file, "wb") as sf:
            pickle.dump(customer_obj, sf)
        os.replace(tmp_file, state_file)
        log_message("system", f"Background state saved to {state_file.name}")
    except Exception as e:
        log_message("system", f"Background state save warning: {e}")

def download_states_from_huggingface(model_id: str = "default") -> bool:
    """
    Downloads customer states and global seed for model_id from Hugging Face private dataset repo.
    Returns True if successfully retrieved, False otherwise.
    """
    token: str = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("GITHUB_PAT") or ""
    if not token:
        log_message("system", "[HF Hydration] No HF_TOKEN provided. Skipping Hugging Face download.")
        return False

    log_message("system", f"[HF Hydration] Downloading states for model '{model_id}' from dataset '{HF_REPO_ID}'...")
    try:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi(token=token)
        
        subfolder = f"models/{model_id}"
        local_dir = Path("hf_download_temp")
        if local_dir.exists():
            shutil.rmtree(local_dir, ignore_errors=True)
            
        downloaded_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=token,
            allow_patterns=f"{subfolder}/*",
            local_dir=str(local_dir)
        )
        
        model_dir = local_dir / subfolder
        restored_count = 0
        if model_dir.exists():
            states_src = model_dir / "states"
            if states_src.exists():
                for bin_file in states_src.glob("*.bin"):
                    shutil.copy2(bin_file, STATES_DIR / bin_file.name)
                    restored_count += 1
            
            global_src = model_dir / "global_cache"
            if global_src.exists():
                for bin_file in global_src.glob("*.bin"):
                    shutil.copy2(bin_file, GLOBAL_CACHE_DIR / bin_file.name)
                    
        shutil.rmtree(local_dir, ignore_errors=True)
        log_message("system", f"[HF Hydration] Restored {restored_count} customer states for model '{model_id}'.")
        return True
    except Exception as err:
        log_message("system", f"[HF Hydration] Warning: Hugging Face state download failed: {err}")
        return False

def upload_states_to_huggingface(model_id: str = "default") -> bool:
    """
    Uploads all local customer states and global seed files for model_id to Hugging Face private dataset repo.
    Returns True if successfully uploaded, False otherwise.
    """
    token: str = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("GITHUB_PAT") or ""
    if not token:
        log_message("system", "[HF Sync] No HF_TOKEN provided. Skipping Hugging Face upload.")
        return False

    log_message("system", f"[HF Sync] Uploading active states for model '{model_id}' to dataset '{HF_REPO_ID}'...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        
        staging_dir = Path("hf_upload_staging") / "models" / model_id
        if staging_dir.exists():
            shutil.rmtree(staging_dir.parent.parent, ignore_errors=True)
            
        staging_states = staging_dir / "states"
        staging_global = staging_dir / "global_cache"
        staging_states.mkdir(parents=True, exist_ok=True)
        staging_global.mkdir(parents=True, exist_ok=True)
        
        for phone_key, customer_obj in list(_ram_states_cache.items()):
            state_file = STATES_DIR / f"{phone_key}.bin"
            save_state_bg(state_file, customer_obj)
            
        state_count = 0
        for bin_file in STATES_DIR.glob("*.bin"):
            shutil.copy2(bin_file, staging_states / bin_file.name)
            state_count += 1
            
        for bin_file in GLOBAL_CACHE_DIR.glob("*.bin"):
            shutil.copy2(bin_file, staging_global / bin_file.name)
            
        api.upload_folder(
            folder_path=str(staging_dir),
            path_in_repo=f"models/{model_id}",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"Sync {state_count} customer states for model {model_id}"
        )
        
        shutil.rmtree(staging_dir.parent.parent, ignore_errors=True)
        log_message("system", f"[HF Sync] Successfully uploaded {state_count} customer states for model '{model_id}'.")
        return True
    except Exception as err:
        log_message("system", f"[HF Sync] Warning: Hugging Face upload failed: {err}")
        return False

def clear_huggingface_and_local_states(model_id: str = "default") -> bool:
    """
    Clears local RAM & disk states and deletes remote HF dataset files for model_id
    when a new global KV state update is received from kv_worker.
    """
    _ram_states_cache.clear()
    
    for bin_file in STATES_DIR.glob("*.bin"):
        try:
            bin_file.unlink()
        except Exception:
            pass

    token: str = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("GITHUB_PAT") or ""
    if not token:
        log_message("system", "[HF Clear] No HF_TOKEN provided. Cleared local states only.")
        return True

    log_message("system", f"[HF Clear] Deleting remote HF dataset files for model '{model_id}'...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        try:
            api.delete_folder(
                path_in_repo=f"models/{model_id}",
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"Clear outdated states for model {model_id} upon global KV update"
            )
            log_message("system", f"[HF Clear] Successfully cleared remote HF dataset files for model '{model_id}'.")
        except Exception as delete_err:
            if "404" in str(delete_err) or "Not Found" in str(delete_err):
                log_message("system", f"[HF Clear] Remote HF folder 'models/{model_id}' does not exist yet. Skipping deletion.")
            else:
                log_message("system", f"[HF Clear] Folder delete note: {delete_err}")
        return True
    except Exception as err:
        log_message("system", f"[HF Clear] Warning: Hugging Face clear exception: {err}")
        return False




