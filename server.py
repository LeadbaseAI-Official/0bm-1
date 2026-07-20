import os
import time
import json
import base64
import re
import subprocess
import uvicorn
import threading
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from github import Github, Auth
from github.GithubException import UnknownObjectException

from model import run_model_query, MODEL_CODE

app: FastAPI = FastAPI(title="Local GGUF LLM API Server")

class ChatRequest(BaseModel):
    prompt: str
    jid: Optional[str] = None
    image_base64: Optional[str] = None

class ClearRequest(BaseModel):
    jid: Optional[str] = None

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
        
        # Wait up to 15 seconds to extract the trycloudflare.com URL
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
        
        if url:
            return url
        else:
            print("Failed to extract Cloudflare tunnel URL from tunnel.log.", flush=True)
            return None
    except Exception as ex:
        print(f"Failed to start cloudflared tunnel process: {ex}", flush=True)
        return None

# ---------------------------------------------------------------------------
# GitHub DNS Updater & Dispatcher
# ---------------------------------------------------------------------------
def update_github_dns(pat: str, org: str, public_url: str, repo_name: str) -> None:
    print(f"Connecting to GitHub using PAT to update dynamic DNS registry...", flush=True)
    max_attempts: int = 5
    for attempt in range(1, max_attempts + 1):
        try:
            auth_obj: Auth.Token = Auth.Token(pat)
            g: Github = Github(auth=auth_obj)
            
            # Target dns repository
            target_repo_name: str = "dns"
            full_repo_path: str = f"{org}/{target_repo_name}"
            
            repo = g.get_repo(full_repo_path)
            
            # Get current config.json contents
            try:
                contents = repo.get_contents("config.json")
                config_bytes: bytes = base64.b64decode(contents.content)
                sha: str = contents.sha
                try:
                    config_data: dict = json.loads(config_bytes.decode("utf-8"))
                    print("Successfully loaded existing config.json.", flush=True)
                except Exception as parse_err:
                    print(f"Warning: config.json content was not valid JSON ({parse_err}). Initializing fresh dictionary.", flush=True)
                    config_data = {}
            except UnknownObjectException:
                config_data = {}
                sha = ""
                print("config.json not found in dns repo. Creating a fresh registry.", flush=True)

            # Resolve the sub-dictionary key prefix from repo_name dynamically
            # e.g. "private-model-1" -> "private-model", "2bm-1" -> "2bm", "0bm-1" -> "0bm"
            import re
            match = re.match(r"^(.*?)-\d+$", repo_name)
            if match:
                model_code_to_write = match.group(1)
            else:
                model_code_to_write = MODEL_CODE

            # Set key under the specific model code sub-dictionary
            if model_code_to_write not in config_data:
                config_data[model_code_to_write] = {}
            
            # Keep config_data dict clean: remove old flat key if it exists
            if repo_name in config_data:
                del config_data[repo_name]
                
            # Remove key from other categories dynamically to prevent cross-contamination
            for key in list(config_data.keys()):
                if key != model_code_to_write and isinstance(config_data[key], dict) and repo_name in config_data[key]:
                    del config_data[key][repo_name]
                
            config_data[model_code_to_write][repo_name] = public_url
            updated_json: str = json.dumps(config_data, indent=2)
            
            if sha:
                repo.update_file(
                    path="config.json",
                    message=f"Update {repo_name} endpoint tunnel DNS URL [automated]",
                    content=updated_json,
                    sha=sha
                )
                print(f"config.json updated successfully with key '{repo_name}'.", flush=True)
            else:
                repo.create_file(
                    path="config.json",
                    message=f"Create tunnel DNS registry config.json with key '{repo_name}' [automated]",
                    content=updated_json
                )
                print(f"config.json created successfully with key '{repo_name}'.", flush=True)
            return
        except Exception as e:
            import random
            print(f"Error updating GitHub DNS file (attempt {attempt}/{max_attempts}): {e}", flush=True)
            if attempt < max_attempts:
                sleep_time = random.uniform(2.0, 7.0)
                print(f"Retrying DNS update in {sleep_time:.2f} seconds...", flush=True)
                time.sleep(sleep_time)
            else:
                print("All DNS update attempts failed.", flush=True)

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
    print(f"Graceful shutdown timer started: Server will run for {duration_hours} hours ({duration_seconds} seconds).", flush=True)
    
    time.sleep(duration_seconds)
    
    print("Timer expired. Initiating graceful shutdown and restart...", flush=True)
    
    # 1. Trigger next workflow run
    if pat and repo_name != "test":
        trigger_self_workflow(pat, org, repo_name)
    else:
        print("Local mode or GITHUB_PAT missing, skipping self-dispatch trigger.", flush=True)
        
    # 2. Short wait to allow dispatch request to register
    time.sleep(5)
    
    # 3. Kill cloudflared tunnel
    global tunnel_process
    if tunnel_process:
        try:
            tunnel_process.terminate()
            tunnel_process.wait(timeout=5)
            print("cloudflared tunnel terminated.", flush=True)
        except Exception as te:
            print(f"Error terminating cloudflared: {te}", flush=True)
        
    print("Exiting server process gracefully with code 0.", flush=True)
    os._exit(0)

# ---------------------------------------------------------------------------
# Startup Event
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event() -> None:
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

    # --- Model Warmup & Global Prefix Pre-caching ---
    print("[Warmup] Warming up model weights and building global prefix cache...", flush=True)
    try:
        import re
        import pickle
        from model import get_llm, STATES_DIR
        
        system_path = Path("system.md")
        persona_path = Path("persona.json")
        kb_path = Path("kb.json")
        
        if system_path.exists() and persona_path.exists() and kb_path.exists():
            import json
            from chat_template import format_chat_prompt
            
            system_prompt = system_path.read_text(encoding="utf-8").strip()
            persona_raw = persona_path.read_text(encoding="utf-8").strip()
            kb = kb_path.read_text(encoding="utf-8").strip()
            
            # Strip tool declarations
            clean_system = re.sub(r"## TOOL USE[\s\S]*?(?=##|$)", "", system_prompt)
            
            # Format persona exactly like auto-reply.ts
            try:
                persona_data = json.loads(persona_raw)
                if isinstance(persona_data.get("persona"), dict):
                    persona = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in persona_data["persona"].items())
                else:
                    persona = str(persona_data.get("persona", "none"))
            except Exception:
                persona = "none"
            
            # Format raw prefix exactly like buildPrompt
            prompt_parts = [
                "System Prompt:",
                clean_system,
                "",
                "Persona:",
                persona,
                "",
                "Knowledge Base (Authoritative Facts):",
                kb,
                "",
                "Conversation History:",
                "none",
                "",
                "Current Query:",
                "Hi",
                "",
                "Instruction: Answer the customer's query using only the facts in the Knowledge Base above. Be helpful, polite, and professional.",
                "",
                "Assistant:"
            ]
            prompt_text = "\n".join(prompt_parts)
            formatted_prefix = format_chat_prompt(prompt_text)
            
            llm = get_llm()
            prefix_tokens = llm.tokenize(formatted_prefix.encode("utf-8"))
            
            state_file = STATES_DIR / "global_prefix.state"
            
            # Check if valid cache exists
            cache_valid = False
            if state_file.exists():
                try:
                    with open(state_file, "rb") as sf:
                        raw_data = pickle.load(sf)
                    cached_tokens = raw_data["tokens"]
                    if cached_tokens == prefix_tokens:
                        cache_valid = True
                except Exception:
                    cache_valid = False
            
            if cache_valid:
                print(f"[Warmup] Found valid global prefix cache ({len(prefix_tokens)} tokens). Skipping evaluation.", flush=True)
            else:
                print(f"[Warmup] Cache invalid or missing. Compiling global prefix cache ({len(prefix_tokens)} tokens)...", flush=True)
                llm.reset()
                llm.eval(prefix_tokens)
                
                # Write state files
                state_data = llm.save_state()
                with open(state_file, "wb") as sf:
                    pickle.dump({"state": state_data, "tokens": prefix_tokens}, sf)
                print(f"[Warmup] Inline global prefix cache compiled and saved to disk.", flush=True)
            
            print(f"[Warmup] Warmup phase complete.", flush=True)
        else:
            print("[Warmup] Warning: Warmup files system.md/persona.json/kb.json not found on disk. Performing fallback query...", flush=True)
            warmup_res = run_model_query("Hi")
            print(f"[Warmup] Fallback query completed: {warmup_res[:40].strip()!r}", flush=True)
    except Exception as warmup_err:
        print(f"[Warmup] Warning: global prefix caching failed: {warmup_err}", flush=True)

    # Start Cloudflare Quick Tunnel
    public_url: Optional[str] = start_cloudflare_tunnel()
    if public_url:
        print(f"==================================================", flush=True)
        print(f"CLOUDFLARE TUNNEL ESTABLISHED SUCCESSFULLY!", flush=True)
        print(f"Public API Address: {public_url}", flush=True)
        print(f"==================================================", flush=True)
        
        # Write back tunnel DNS to config.json
        if pat:
            update_github_dns(pat, org, public_url, repo_name)
        else:
            print("Warning: GITHUB_PAT not configured. Skipping DNS config.json registration.", flush=True)
    else:
        print("Running server without public tunnel.", flush=True)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/chat")
async def chat(req: ChatRequest) -> dict:
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt parameter is required.")
    
    response_text: str = await run_model_query(req.prompt, req.jid, req.image_base64)
    return {
        "response": response_text,
        "prompt": req.prompt
    }

@app.post("/v1/chat/clear")
async def clear_chat_state(req: ClearRequest) -> dict:
    try:
        from model import STATES_DIR
        if req.jid:
            state_file = STATES_DIR / f"{req.jid}.state"
            if state_file.exists():
                state_file.unlink()
                print(f"[Model] Cleared conversation history cache for JID: {req.jid}", flush=True)
        else:
            for path in STATES_DIR.glob("*.state"):
                if path.name != "global_prefix.state":
                    path.unlink()
            print("[Model] Cleared all JID conversation history caches", flush=True)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, access_log=False)
