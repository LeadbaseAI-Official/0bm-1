import os
import time
import json
import base64
import re
import subprocess
import uvicorn
import threading
import requests
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from github import Github, Auth
from github.GithubException import UnknownObjectException
from contextlib import asynccontextmanager

from model import run_model_query, MODEL_CODE

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
        print(f"cloudflared binary not found or not working: {e}. Running without tunnel.", flush=True)
        return None

    print(f"Starting cloudflared tunnel using: {cmd}", flush=True)
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
        print(f"Failed to start cloudflared tunnel process: {ex}", flush=True)
        return None

# ---------------------------------------------------------------------------
# GitHub DNS Updater & Dispatcher (Via Cloudflare Worker)
# ---------------------------------------------------------------------------
def update_github_dns(pat: str, org: str, public_url: str, repo_name: str) -> None:
    max_attempts: int = 5
    match = re.match(r"^(.*?)-(\d+)$", repo_name)
    superkey = match.group(1) if match else MODEL_CODE
    
    dns_key = f"{superkey}/{repo_name}"
    print(f"Updating dynamic DNS registry via Cloudflare Worker... Key: {dns_key}", flush=True)
    
    for attempt in range(1, max_attempts + 1):
        try:
            payload = {"key": dns_key, "value": public_url}
            res = requests.post("https://dns-manager.aakashmishra2050880.workers.dev/update", json=payload, timeout=10)
            if res.status_code == 200:
                print(f"DNS updated successfully for key '{dns_key}' with URL {public_url}", flush=True)
                return
            else:
                print(f"CF Worker returned status code {res.status_code}: {res.text}", flush=True)
        except Exception as e:
            import random
            print(f"Error updating DNS (attempt {attempt}/{max_attempts}): {e}", flush=True)
            time.sleep(random.uniform(2.0, 5.0))

def trigger_standby_sync(pat: str, org: str, repo_name: str) -> None:
    """
    Stitches all local SSD state files and uploads them to the Standby Server.
    """
    print(f"[Handoff] Gathering local cached states to sync to Standby Server...", flush=True)
    try:
        # Resolve Standby Server URL from public raw config.json
        res_dns = requests.get(f"https://raw.githubusercontent.com/{org}/dns/main/config.json", timeout=10)
        if res_dns.status_code != 200:
            print(f"[Handoff] Failed to fetch DNS registry from GitHub raw (Code: {res_dns.status_code})", flush=True)
            return
            
        config_data = res_dns.json()
        standby_url: Optional[str] = config_data.get("standby", {}).get("standby-server")
        
        if not standby_url:
            print("[Handoff] Standby Server URL not found in registry. Handoff sync aborted.", flush=True)
            return

        # 1. Gather Client Globals
        globals_list = []
        global_cache_dir = Path("global_cache")
        if global_cache_dir.exists():
            for path in global_cache_dir.glob("*.bin"):
                client_id = path.stem
                with open(path, "rb") as f:
                    file_bytes = f.read()
                b64_str = base64.b64encode(file_bytes).decode("utf-8")
                globals_list.append({"client_id": client_id, "state_bytes_base64": b64_str})

        # 2. Gather Phone States
        phones_list = []
        from model import STATES_DIR
        if STATES_DIR.exists():
            for path in STATES_DIR.glob("*.bin"):
                phone_num = path.stem
                with open(path, "rb") as f:
                    file_bytes = f.read()
                b64_str = base64.b64encode(file_bytes).decode("utf-8")
                phones_list.append({"phone_number": phone_num, "state_bytes_base64": b64_str})

        # Send payload to standby server
        payload = {
            "model_id": repo_name,
            "globals": globals_list,
            "phones": phones_list
        }
        res = requests.post(f"{standby_url.rstrip('/')}/v1/sync", json=payload, timeout=30)
        if res.status_code == 200:
            print("[Handoff] Handoff sync payload uploaded to Standby Server successfully.", flush=True)
        else:
            print(f"[Handoff] Standby Server returned error {res.status_code}: {res.text}", flush=True)
    except Exception as e:
        print(f"[Handoff] Failed to sync to Standby Server: {e}", flush=True)

def recover_states_from_standby(pat: str, org: str, repo_name: str) -> None:
    """
    Queries the Standby Server on boot to pull back active states and restore DNS.
    """
    print(f"[Startup Recovery] Recovering states from Standby Server for {repo_name}...", flush=True)
    try:
        res_dns = requests.get(f"https://raw.githubusercontent.com/{org}/dns/main/config.json", timeout=10)
        if res_dns.status_code != 200:
            print(f"[Startup Recovery] Failed to fetch DNS registry from GitHub raw (Code: {res_dns.status_code})", flush=True)
            return
            
        config_data = res_dns.json()
        standby_url: Optional[str] = config_data.get("standby", {}).get("standby-server")
        
        if not standby_url:
            print("[Startup Recovery] Standby Server URL not found. Running clean start.", flush=True)
            return

        res = requests.get(f"{standby_url.rstrip('/')}/v1/get-sync?model_id={repo_name}", timeout=30)
        if res.status_code == 200:
            data = res.json()
            
            # Write global caches
            global_cache_dir = Path("global_cache")
            global_cache_dir.mkdir(exist_ok=True)
            for item in data.get("globals", []):
                client_id = item["client_id"]
                file_bytes = base64.b64decode(item["state_bytes_base64"])
                with open(global_cache_dir / f"{client_id}.bin", "wb") as f:
                    f.write(file_bytes)
                    
            # Write phone convo caches
            from model import STATES_DIR
            STATES_DIR.mkdir(exist_ok=True)
            for item in data.get("phones", []):
                phone_num = item["phone_number"]
                file_bytes = base64.b64decode(item["state_bytes_base64"])
                with open(STATES_DIR / f"{phone_num}.bin", "wb") as f:
                    f.write(file_bytes)
            print(f"[Startup Recovery] Restored {len(data.get('phones', []))} conversation states from Standby Server.", flush=True)
        else:
            print(f"[Startup Recovery] Standby Server returned {res.status_code}. No backup found.", flush=True)
    except Exception as e:
        print(f"[Startup Recovery] Warning: Recovery failed: {e}", flush=True)

def trigger_self_workflow(pat: str, org: str, repo_name: str) -> None:
    print(f"Triggering self workflow dispatch for repository {repo_name}...", flush=True)
    try:
        auth_obj: Auth.Token = Auth.Token(pat)
        g: Github = Github(auth=auth_obj)
        repo = g.get_repo(f"{org}/{repo_name}")
        default_branch: str = repo.default_branch
        
        # Trigger standard workflow.yml on the default branch
        wf = repo.get_workflow("workflow.yml")
        wf.create_dispatch(default_branch)
        print("Self workflow dispatch triggered successfully.", flush=True)
    except Exception as e:
        print(f"Failed to trigger self workflow: {e}", flush=True)

def shutdown_timer(pat: str, org: str, repo_name: str, duration_hours: float) -> None:
    duration_seconds: float = duration_hours * 3600
    
    # Calculate handoff sync time (15 minutes before shutdown)
    sync_lead_time = 15 * 60
    sleep_first = max(0.0, duration_seconds - sync_lead_time)
    
    print(f"Graceful shutdown timer started: Server will run for {duration_hours} hours. Handoff sync in {sleep_first / 60:.1f} minutes.", flush=True)
    time.sleep(sleep_first)
    
    # 1. Trigger handoff sync to standby
    if pat:
        trigger_standby_sync(pat, org, repo_name)
        
    # Wait remaining 15 minutes
    print(f"[Handoff] Waiting remaining 15 minutes before runner shutdown...", flush=True)
    time.sleep(sync_lead_time)
    
    print("Timer expired. Initiating graceful shutdown and restart...", flush=True)
    
    # 2. Trigger next workflow run
    if pat and repo_name != "test":
        trigger_self_workflow(pat, org, repo_name)
        
    time.sleep(5)
    
    # 3. Kill cloudflared tunnel
    global tunnel_process
    if tunnel_process:
        try:
            tunnel_process.terminate()
        except Exception:
            pass
        
    print("Exiting server process gracefully with code 0.", flush=True)
    os._exit(0)

# ---------------------------------------------------------------------------
# Lifespan Events Handler (Startup & Shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    pat: str = os.getenv("GITHUB_PAT", "")
    org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")

    # Resolve repo name from standard environment variable
    repo_full: str = os.getenv("GITHUB_REPOSITORY", "")
    repo_name: str = repo_full.split("/")[-1] if "/" in repo_full else "test"

    # Start the shutdown timer thread
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

    # --- Restore state from Standby Server if recovering ---
    if pat and repo_name != "test":
        recover_states_from_standby(pat, org, repo_name)

    # --- Model Warmup ---
    print("[Warmup] Warming up model weights...", flush=True)
    try:
        from model import get_llm
        get_llm()
        
        # Create global_cache directory
        global_cache_dir = Path("global_cache")
        global_cache_dir.mkdir(exist_ok=True)
        print("[Warmup] Model initialized successfully.", flush=True)
    except Exception as warmup_err:
        print(f"[Warmup] Warning: model warmup failed: {warmup_err}", flush=True)

    # Start Cloudflare Quick Tunnel
    public_url: Optional[str] = start_cloudflare_tunnel()
    if public_url:
        print(f"==================================================", flush=True)
        print(f"CLOUDFLARE TUNNEL ESTABLISHED SUCCESSFULLY!", flush=True)
        print(f"Public API Address: {public_url}", flush=True)
        print(f"==================================================", flush=True)
        
        # Write back tunnel DNS to config.json (Re-registers itself to original slot)
        if pat:
            update_github_dns(pat, org, public_url, repo_name)
        else:
            print("Warning: GITHUB_PAT not configured. Skipping DNS config.json registration.", flush=True)
    else:
        print("Running server without public tunnel.", flush=True)
        
    yield
    # No custom shutdown tasks needed outside the daemon shutdown loop

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
    """
    Receives and caches a client's global pre-compiled prefix state.
    """
    try:
        global_cache_dir = Path("global_cache")
        global_cache_dir.mkdir(exist_ok=True)
        
        file_bytes = base64.b64decode(req.state_bytes_base64)
        state_file = global_cache_dir / f"{req.client_id}.bin"
        
        with open(state_file, "wb") as f:
            f.write(file_bytes)
            
        print(f"[Model] Successfully updated global cache for client: {req.client_id}", flush=True)
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
                print(f"[Model] Cleared conversation history cache for phone: {req.phone_number}", flush=True)
        else:
            for path in STATES_DIR.glob("*.bin"):
                path.unlink()
            print("[Model] Cleared all phone conversation history caches", flush=True)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, access_log=False)
