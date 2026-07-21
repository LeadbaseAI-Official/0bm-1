import os
import time
import json
import base64
import re
import subprocess
import uvicorn
import threading
import requests
import datetime
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from github import Github, Auth
from github.GithubException import UnknownObjectException
from contextlib import asynccontextmanager

from model import run_model_query, MODEL_CODE, log_message # type: ignore

class ChatRequest(BaseModel):
    prompt: str
    client_id: Optional[str] = None
    phone_number: Optional[str] = None
    image_base64: Optional[str] = None

class ClearRequest(BaseModel):
    phone_number: Optional[str] = None

class GlobalUpdateItem(BaseModel):
    client_id: str
    state_bytes_base64: str

# Global handle for cloudflared process
tunnel_process: Optional[subprocess.Popen] = None

# ---------------------------------------------------------------------------
# Cloudflare Tunnel Manager
# ---------------------------------------------------------------------------
def start_cloudflare_tunnel() -> Optional[str]:
    global tunnel_process
    cmd: str = "./cloudflared" if os.path.exists("./cloudflared") else "cloudflared"
    
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
    except Exception as e:
        log_message("system", f"cloudflared binary not found or not working: {e}. Running without tunnel.")
        return None

    log_message("system", f"Starting cloudflared tunnel using: {cmd}")
    try:
        log_file = open("tunnel.log", "w")
        tunnel_process = subprocess.Popen(
            [cmd, "tunnel", "--url", "http://localhost:8000"],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        url: Optional[str] = None
        for i in range(15):
            time.sleep(1)
            if os.path.exists("tunnel.log"):
                with open("tunnel.log", "r") as f:
                    content: str = f.read()
                    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                    if match:
                        url = match.group(0)
                        break
        log_file.close()
        return url
    except Exception as ex:
        log_message("system", f"Failed to start cloudflared tunnel process: {ex}")
        return None

SUPERKEY = "0bm"

# ---------------------------------------------------------------------------
# GitHub DNS Updater & Dispatcher (Via Cloudflare Worker)
# ---------------------------------------------------------------------------
def update_github_dns(pat: str, org: str, public_url: str, repo_name: str) -> None:
    max_attempts: int = 5
    dns_key = f"{SUPERKEY}/{repo_name}"
    log_message("system", f"Updating dynamic DNS registry via Cloudflare Worker... Key: {dns_key}")
    
    for attempt in range(1, max_attempts + 1):
        try:
            payload = {"key": dns_key, "value": public_url}
            res = requests.post("https://dns-manager.aakashmishra2050880.workers.dev/update", json=payload, timeout=10)
            if res.status_code == 200:
                log_message("system", f"DNS updated successfully for key '{dns_key}' with URL {public_url}")
                return
            else:
                log_message("system", f"CF Worker returned status code {res.status_code}: {res.text}")
        except Exception as e:
            import random
            log_message("system", f"Error updating DNS (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(random.uniform(2.0, 5.0))

def trigger_standby_sync(pat: str, org: str, repo_name: str) -> None:
    log_message("system", "[Handoff] Gathering local cached states to sync to Standby Server...")
    try:
        res_dns = requests.get(f"https://raw.githubusercontent.com/{org}/dns/main/config.json", timeout=10)
        if res_dns.status_code != 200:
            log_message("system", f"[Handoff] Failed to fetch DNS registry from GitHub raw (Code: {res_dns.status_code})")
            return
            
        config_data = res_dns.json()
        standby_url: Optional[str] = config_data.get("standby-server", {}).get("standby-server")
        
        if not standby_url:
            log_message("system", "[Handoff] Standby Server URL not found in registry. Handoff sync aborted.")
            return

        import gzip

        # 1. Gather Client Globals
        globals_list = []
        global_cache_dir = Path("global_cache")
        if global_cache_dir.exists():
            for path in global_cache_dir.glob("*.bin"):
                client_id = path.stem
                with open(path, "rb") as f:
                    file_bytes = f.read()
                compressed_bytes = gzip.compress(file_bytes)
                b64_str = base64.b64encode(compressed_bytes).decode("utf-8")
                globals_list.append({"client_id": client_id, "state_bytes_base64": b64_str})

        # 2. Gather Phone States
        phones_list = []
        from model import STATES_DIR
        if STATES_DIR.exists():
            for path in STATES_DIR.glob("*.bin"):
                phone_num = path.stem
                with open(path, "rb") as f:
                    file_bytes = f.read()
                compressed_bytes = gzip.compress(file_bytes)
                b64_str = base64.b64encode(compressed_bytes).decode("utf-8")
                phones_list.append({"phone_number": phone_num, "state_bytes_base64": b64_str})

        payload = {
            "model_id": repo_name,
            "globals": globals_list,
            "phones": phones_list
        }
        res = requests.post(f"{standby_url.rstrip('/')}/v1/sync", json=payload, timeout=30)
        if res.status_code == 200:
            log_message("system", "[Handoff] Handoff sync payload uploaded to Standby Server successfully.")
        else:
            log_message("system", f"[Handoff] Standby Server returned error {res.status_code}: {res.text}")
    except Exception as e:
        log_message("system", f"[Handoff] Failed to sync to Standby Server: {e}")

def recover_states_from_standby(pat: str, org: str, repo_name: str) -> None:
    log_message("system", f"[Startup Recovery] Recovering states from Standby Server for {repo_name}...")
    try:
        res_dns = requests.get(f"https://raw.githubusercontent.com/{org}/dns/main/config.json", timeout=10)
        if res_dns.status_code != 200:
            log_message("system", f"[Startup Recovery] Failed to fetch DNS registry from GitHub raw (Code: {res_dns.status_code})")
            return
            
        config_data = res_dns.json()
        standby_url: Optional[str] = config_data.get("standby-server", {}).get("standby-server")
        
        if not standby_url:
            log_message("system", "[Startup Recovery] Standby Server URL not found. Running clean start.")
            return

        res = requests.get(f"{standby_url.rstrip('/')}/v1/get-sync?model_id={repo_name}", timeout=30)
        if res.status_code == 200:
            data = res.json()
            import gzip
            
            # Write global caches
            global_cache_dir = Path("global_cache")
            global_cache_dir.mkdir(exist_ok=True)
            for item in data.get("globals", []):
                client_id = item["client_id"]
                file_bytes = base64.b64decode(item["state_bytes_base64"])
                try:
                    decompressed = gzip.decompress(file_bytes)
                except Exception:
                    decompressed = file_bytes
                with open(global_cache_dir / f"{client_id}.bin", "wb") as f:
                    f.write(decompressed)
                    
            # Write phone convo caches
            from model import STATES_DIR
            STATES_DIR.mkdir(exist_ok=True)
            for item in data.get("phones", []):
                phone_num = item["phone_number"]
                file_bytes = base64.b64decode(item["state_bytes_base64"])
                try:
                    decompressed = gzip.decompress(file_bytes)
                except Exception:
                    decompressed = file_bytes
                with open(STATES_DIR / f"{phone_num}.bin", "wb") as f:
                    f.write(decompressed)
            log_message("system", f"[Startup Recovery] Restored {len(data.get('phones', []))} conversation states from Standby Server.")
            log_message("system", f"[Startup Recovery] Restored {len(data.get('phones', []))} conversation states from Standby Server.")
            return True
        else:
            log_message("system", f"[Startup Recovery] Standby Server returned {res.status_code}. No backup found.")
    except Exception as e:
        log_message("system", f"[Startup Recovery] Warning: Recovery failed: {e}")
    return False

def recover_globals_from_redis(org: str) -> None:
    log_message("system", "[Startup Recovery] Standby recovery missed. Pulling global states from active Redis relay...")
    try:
        res_dns = requests.get(f"https://raw.githubusercontent.com/{org}/dns/main/config.json", timeout=10)
        if res_dns.status_code != 200:
            log_message("system", f"[Startup Recovery] Failed to fetch DNS registry from GitHub raw (Code: {res_dns.status_code})")
            return
            
        config_data = res_dns.json()
        redis_url: Optional[str] = config_data.get("redis-worker", {}).get("active")
        
        if not redis_url:
            log_message("system", "[Startup Recovery] No active Redis worker registered in DNS config. Running clean start.")
            return
            
        res = requests.get(f"{redis_url.rstrip('/')}/v1/get-all-global-states", timeout=30)
        if res.status_code == 200:
            data = res.json()
            restored_count = 0
            
            global_cache_dir = Path("global_cache")
            global_cache_dir.mkdir(exist_ok=True)
            
            import gzip
            for key, val in data.items():
                if key.startswith("global:"):
                    client_id = key.replace("global:", "", 1)
                    file_bytes = base64.b64decode(val)
                    try:
                        decompressed = gzip.decompress(file_bytes)
                    except Exception:
                        decompressed = file_bytes
                        
                    with open(global_cache_dir / f"{client_id}.bin", "wb") as f:
                        f.write(decompressed)
                    restored_count += 1
            log_message("system", f"[Startup Recovery] Restored {restored_count} client global states from active Redis.")
        else:
            log_message("system", f"[Startup Recovery] Redis returned error code {res.status_code}")
    except Exception as e:
        log_message("system", f"[Startup Recovery] Failed to pull from Redis: {e}")

def trigger_self_workflow(pat: str, org: str, repo_name: str) -> None:
    log_message("system", f"Triggering self workflow dispatch for repository {repo_name}...")
    try:
        auth_obj: Auth.Token = Auth.Token(pat)
        g: Github = Github(auth=auth_obj)
        repo = g.get_repo(f"{org}/{repo_name}")
        default_branch: str = repo.default_branch
        
        wf = repo.get_workflow("workflow.yml")
        wf.create_dispatch(default_branch)
        log_message("system", "Self workflow dispatch triggered successfully.")
    except Exception as e:
        log_message("system", f"Failed to trigger self workflow: {e}")

def shutdown_timer(pat: str, org: str, repo_name: str, duration_hours: float) -> None:
    duration_seconds: float = duration_hours * 3600
    
    sync_lead_time = 15 * 60
    sleep_first = max(0.0, duration_seconds - sync_lead_time)
    
    log_message("system", f"Graceful shutdown timer started: Server will run for {duration_hours} hours. Handoff sync in {sleep_first / 60:.1f} minutes.")
    time.sleep(sleep_first)
    
    if pat:
        trigger_standby_sync(pat, org, repo_name)
        
    log_message("system", "[Handoff] Waiting remaining 15 minutes before runner shutdown...")
    time.sleep(sync_lead_time)
    
    log_message("system", "Timer expired. Initiating graceful shutdown and restart...")
    
    if pat and repo_name != "test":
        trigger_self_workflow(pat, org, repo_name)
        
    time.sleep(5)
    
    log_message("system", "Exiting server process gracefully with code 0.")
    os._exit(0)

def lru_eviction_check() -> None:
    """
    Periodic task running every 30 minutes to clean conversation cache pools.
    Evicts phone states older than 30 minutes to the active redis-worker.
    Also enforces the maximum capacity limit of 15 active phone caches.
    """
    from model import STATES_DIR
    org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")
    
    while True:
        time.sleep(1800) # Check every 30 minutes (1800 seconds)
        log_message("system", "Starting periodic LRU conversation state eviction check...")
        
        try:
            # 1. Gather all local session cache files
            state_files = list(STATES_DIR.glob("*.bin"))
            state_files.sort(key=lambda x: x.stat().st_mtime) # Sort by oldest modification time
            
            # Fetch active redis-worker URL
            res_dns = requests.get(f"https://raw.githubusercontent.com/{org}/dns/main/config.json", timeout=10)
            redis_url = None
            if res_dns.status_code == 200:
                redis_url = res_dns.json().get("redis-worker", {}).get("active")
            
            # 2. Identify and evict idle sessions (> 30 minutes old)
            now = time.time()
            active_sessions = []
            for path in state_files:
                age_seconds = now - path.stat().st_mtime
                if age_seconds > 1800:
                    phone_num = path.stem
                    log_message("system", f"Evicting idle conversation state for {phone_num} (Age: {age_seconds / 60:.1f} minutes)...")
                    
                    if redis_url:
                        try:
                            import gzip
                            with open(path, "rb") as f:
                                file_bytes = f.read()
                            compressed = gzip.compress(file_bytes)
                            b64_str = base64.b64encode(compressed).decode("utf-8")
                            
                            payload = {
                                "key": f"state:{phone_num}",
                                "value": b64_str
                            }
                            res = requests.post(f"{redis_url.rstrip('/')}/add", json=payload, timeout=20)
                            if res.status_code == 200:
                                path.unlink()
                                log_message("system", f"Successfully evicting state:{phone_num} to Redis and unlinked local cache.")
                                
                                # Notify frontend to clear JID sticky route
                                try:
                                    dns_data = res_dns.json()
                                    frontend_url = dns_data.get("frontend", {}).get("active") or "http://localhost:3000"
                                    requests.post(f"{frontend_url.rstrip('/')}/api/autoreply/routing/evict", json={"phone_number": phone_num}, timeout=5)
                                except Exception as f_err:
                                    log_message("system", f"Failed to notify frontend eviction for {phone_num}: {f_err}")
                            else:
                                log_message("system", f"Failed to evict {phone_num} to Redis (Returned {res.status_code})")
                        except Exception as upload_err:
                            log_message("system", f"Error uploading evicted state for {phone_num}: {upload_err}")
                    else:
                        log_message("system", f"No active redis-worker URL registered. Retaining local cache for {phone_num}.")
                else:
                    active_sessions.append(path)
            
            # 3. Enforce maximum capacity constraint (15 active files)
            # If still exceeding 15 files, evict the oldest remaining files immediately
            if len(active_sessions) > 15:
                over_limit_count = len(active_sessions) - 15
                log_message("system", f"Runner capacity limit exceeded ({len(active_sessions)}/15). Forcing eviction of {over_limit_count} oldest active sessions...")
                for i in range(over_limit_count):
                    path = active_sessions[i]
                    phone_num = path.stem
                    log_message("system", f"Forcing eviction of least recently used active session: {phone_num}")
                    
                    if redis_url:
                        try:
                            import gzip
                            with open(path, "rb") as f:
                                file_bytes = f.read()
                            compressed = gzip.compress(file_bytes)
                            b64_str = base64.b64encode(compressed).decode("utf-8")
                            
                            payload = {
                                "key": f"state:{phone_num}",
                                "value": b64_str
                            }
                            res = requests.post(f"{redis_url.rstrip('/')}/add", json=payload, timeout=20)
                            if res.status_code == 200:
                                path.unlink()
                                log_message("system", f"Forced eviction success: unlinked local {phone_num}.bin")
                                
                                # Notify frontend to clear JID sticky route
                                try:
                                    dns_data = res_dns.json()
                                    frontend_url = dns_data.get("frontend", {}).get("active") or "http://localhost:3000"
                                    requests.post(f"{frontend_url.rstrip('/')}/api/autoreply/routing/evict", json={"phone_number": phone_num}, timeout=5)
                                except Exception as f_err:
                                    log_message("system", f"Failed to notify frontend eviction for {phone_num}: {f_err}")
                        except Exception as force_err:
                            log_message("system", f"Error during forced eviction of {phone_num}: {force_err}")
                            
        except Exception as lru_err:
            log_message("system", f"Error during LRU eviction check loop: {lru_err}")

# ---------------------------------------------------------------------------
# Lifespan Events Handler (Startup & Shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    pat: str = os.getenv("GITHUB_PAT", "")
    org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")
    repo_full: str = os.getenv("GITHUB_REPOSITORY", "")
    repo_name: str = repo_full.split("/")[-1] if "/" in repo_full else "test"

    duration_str: str = os.getenv("RUN_DURATION_HOURS", "4.0")
    try:
        duration_hours: float = float(duration_str)
    except ValueError:
        duration_hours = 4.0

    t: threading.Thread = threading.Thread(
        target=shutdown_timer,
        args=(pat, org, repo_name, duration_hours),
        daemon=True
    )
    t.start()

    # Start LRU eviction checking loop thread
    lru_thread = threading.Thread(target=lru_eviction_check, daemon=True)
    lru_thread.start()

    if pat and repo_name != "test":
        recovered = recover_states_from_standby(pat, org, repo_name)
        if not recovered:
            recover_globals_from_redis(org)
    elif repo_name == "test":
        recover_globals_from_redis(org)

    log_message("system", "Warming up model weights...")
    try:
        from model import get_llm
        get_llm()
        
        global_cache_dir = Path("global_cache")
        global_cache_dir.mkdir(exist_ok=True)
        log_message("system", "Model initialized successfully.")
    except Exception as warmup_err:
        log_message("system", f"Warning: model warmup failed: {warmup_err}")

    public_url: Optional[str] = start_cloudflare_tunnel()
    if public_url:
        log_message("system", f"CLOUDFLARE TUNNEL ESTABLISHED SUCCESSFULLY! Address: {public_url}")
        
        if pat:
            update_github_dns(pat, org, public_url, repo_name)
        else:
            log_message("system", "Warning: GITHUB_PAT not configured. Skipping DNS registration.")
    else:
        log_message("system", "Running server without public tunnel.")
        
    yield

app = FastAPI(title="Local GGUF LLM API Server", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/chat")
async def chat(req: ChatRequest) -> dict:
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt parameter is required.")
    
    response_text: str = await run_model_query(req.prompt, req.client_id, req.phone_number, req.image_base64)
    return {
        "response": response_text,
        "prompt": req.prompt
    }

@app.post("/v1/global-update")
def receive_global_update(req: GlobalUpdateItem) -> dict:
    try:
        global_cache_dir = Path("global_cache")
        global_cache_dir.mkdir(exist_ok=True)
        
        file_bytes = base64.b64decode(req.state_bytes_base64)
        
        import gzip
        try:
            decompressed_bytes = gzip.decompress(file_bytes)
            log_message("system", f"Decompressed global state from {len(file_bytes)} to {len(decompressed_bytes)} bytes.")
        except Exception:
            decompressed_bytes = file_bytes
            
        state_file = global_cache_dir / f"{req.client_id}.bin"
        
        with open(state_file, "wb") as f:
            f.write(decompressed_bytes)
            
        log_message("system", f"Successfully updated global cache for client: {req.client_id}")
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/clear")
async def clear_chat_state(req: ClearRequest) -> dict:
    try:
        from model import STATES_DIR
        if req.phone_number:
            state_file = STATES_DIR / f"{req.phone_number}.bin"
            if state_file.exists():
                state_file.unlink()
                log_message("system", f"Cleared conversation history cache for phone: {req.phone_number}")
        else:
            for path in STATES_DIR.glob("*.bin"):
                path.unlink()
            log_message("system", "Cleared all phone conversation history caches")
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, access_log=False)
