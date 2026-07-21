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

from model import run_model_query, MODEL_CODE, log_message

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
        else:
            log_message("system", f"[Startup Recovery] Standby Server returned {res.status_code}. No backup found.")
    except Exception as e:
        log_message("system", f"[Startup Recovery] Warning: Recovery failed: {e}")

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
    
    global tunnel_process
    if tunnel_process:
        try:
            tunnel_process.terminate()
        except Exception:
            pass
        
    log_message("system", "Exiting server process gracefully with code 0.")
    os._exit(0)

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

    if pat and repo_name != "test":
        recover_states_from_standby(pat, org, repo_name)

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
