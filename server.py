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
import pickle
import gzip
from typing import Optional, Dict, Any, List

from pathlib import Path
from fastapi import FastAPI, HTTPException, Response, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager

from model import (
    run_model_query, MODEL_CODE, log_message, get_llm,
    run_5_30_lifecycle_timer, queue_global_rebuild,
    download_states_from_huggingface, upload_states_to_huggingface,
    clear_huggingface_and_local_states, ensure_input_ids_capacity,
    STATES_DIR, GLOBAL_CACHE_DIR
)


from model.engine import _eval_lock
from model.cache_manager import _ram_states_cache, save_state_bg
from model.lifecycle import is_accepting_requests

class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    user_question: Optional[str] = None
    phone_number: str

class ClearRequest(BaseModel):
    phone_number: str  # number string or "all"

class StateUpdateItem(BaseModel):
    client_id: str
    state_bytes_base64: str
    rebuilt_customer_states: Optional[Dict[str, str]] = None

def send_kv_complete_to_redis(client_id: str) -> bool:
    """
    Writes kv_status:{client_id} = "complete" to Upstash Redis.
    The wa-gateway runner polls this key every 4s and drains the
    fallback message queue when it sees "complete".
    Replaces the old HTTP webhook to walone.vercel.app/api/knowledge/kv-complete.
    """
    upstash_url: str = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
    upstash_token: str = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()

    if not upstash_url or not upstash_token:
        log_message("SIGNAL", f"❌ Upstash Redis env vars missing. Cannot send kv_complete for client '{client_id}'.")
        return False

    redis_key: str = f"kv_status:{client_id}"
    log_message("SIGNAL", f"📡 Writing kv_status=complete to Redis for client '{client_id}' (key: {redis_key})...")

    try:
        res = requests.post(
            upstash_url,
            json=["SET", redis_key, "complete"],
            headers={
                "Authorization": f"Bearer {upstash_token}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        if res.status_code == 200:
            log_message("SIGNAL", f"✅ kv_status=complete written to Redis for client '{client_id}'.")
            return True
        else:
            log_message("SIGNAL", f"⚠️ Redis write returned HTTP {res.status_code}: {res.text}")
            return False
    except Exception as err:
        log_message("SIGNAL", f"❌ Failed to write kv_complete to Redis for client '{client_id}': {err}")
        return False






class SummarizeRequest(BaseModel):
    client_id: Optional[str] = None

class GlobalRecompileRequest(BaseModel):
    client_id: Optional[str] = None
    system_prompt: str
    kb_content: str
    persona: Optional[Any] = None
    webhook_url: str

def extract_customer_summary(customer_key: str, state_file_path: Path) -> str:
    """
    Loads customer's existing binary KV state (.bin) and generates a 100-150 word summary of customer details.
    """
    summary_prompt_text = (
        "<|im_start|>user\n"
        "Summarize all key customer details, name, preferences, budget, "
        "and active inquiries from this conversation in 100-150 words:<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    llm = get_llm()
    with _eval_lock:
        try:
            if customer_key in _ram_states_cache:
                ram_obj = _ram_states_cache[customer_key]
                llm.load_state(ram_obj["state"])
            elif state_file_path.exists():
                with open(state_file_path, "rb") as f:
                    cust_obj = pickle.load(f)
                if isinstance(cust_obj, dict) and "state" in cust_obj:
                    llm.load_state(cust_obj["state"])
            else:
                return ""

            ensure_input_ids_capacity(llm)
            summary_tokens = llm.tokenize(summary_prompt_text.encode("utf-8"), add_bos=False)
            llm.eval(summary_tokens)

            stop_token_ids = set()
            for s_str in ["<|im_end|>", "<|im_start|>", "</|im_start|>", "<|endoftext|>"]:
                try:
                    s_toks = llm.tokenize(s_str.encode("utf-8"), add_bos=False, special=True)
                    stop_token_ids.update(s_toks)
                except Exception:
                    pass

            text_chunks: List[str] = []
            for step in range(200):
                tok_id = llm.sample(temp=0.7, top_k=40, top_p=0.9)
                if tok_id in stop_token_ids:
                    break
                piece = llm.detokenize([tok_id]).decode("utf-8", errors="ignore")
                text_chunks.append(piece)
                llm.eval([tok_id])

            raw_summary: str = "".join(text_chunks).strip()
            cleaned_summary: str = re.sub(r'<think>[\s\S]*?</think>', '', raw_summary, flags=re.IGNORECASE)
            cleaned_summary = re.split(r'</?\|?im_(?:start|end)[\|>\}]?', cleaned_summary, flags=re.IGNORECASE)[0].strip()
            return cleaned_summary or f"Customer {customer_key} active conversation session."
        except Exception as ex:
            log_message("summary", f"Error summarizing customer {customer_key}: {ex}")
            return f"Customer {customer_key} active conversation session."


def prefill_and_save_kv_state(
    customer_key: str,
    system_prompt: str,
    kb_content: str,
    persona: Any,
    summary_text: str
) -> bool:
    """
    Runs prefill pass for [NEW_SYS + NEW_KB + PERSONA + SUMMARY] without generating text
    and exports the compiled binary KV state to states/{customer_key}.bin and RAM LRU.
    """
    persona_str = json.dumps(persona) if isinstance(persona, (dict, list)) else str(persona or "")
    combined_prefix = (
        f"<|im_start|>system\n"
        f"System Prompt:\n{system_prompt.strip()}\n\n"
        f"Knowledge Base:\n{kb_content.strip()}\n\n"
        f"Persona:\n{persona_str.strip()}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Customer Context Summary:\n{summary_text.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"Understood. I have loaded the customer context.<|im_end|>\n"
    )

    llm = get_llm()
    tokens = llm.tokenize(combined_prefix.encode("utf-8"))
    output_bin_path = STATES_DIR / f"{customer_key}.bin"

    with _eval_lock:
        try:
            llm.reset()
            llm.eval(tokens)
            state_obj = llm.save_state()
            actual_n = llm.n_tokens
            full_tokens = llm.input_ids.tolist()[:actual_n]

            customer_obj = {
                "customer_key": customer_key,
                "state": state_obj,
                "tokens": full_tokens,
                "history": [
                    {"role": "user", "content": f"[Context Summary]\n{summary_text}"},
                    {"role": "assistant", "content": "Understood. I have loaded the customer context."}
                ],
                "msg_count": 2
            }

            _ram_states_cache[customer_key] = customer_obj
            _ram_states_cache.move_to_end(customer_key)
            save_state_bg(output_bin_path, customer_obj)
            log_message("compile", f"Successfully compiled new KV state for {customer_key} -> {output_bin_path.name}")
            return True
        except Exception as ex:
            log_message("compile", f"Error prefilling KV state for {customer_key}: {ex}")
            return False

def run_full_recompilation_pipeline(payload: GlobalRecompileRequest) -> None:
    """
    Background worker pipeline:
    1. Summarizes all existing active customer states.
    2. Compiles brand-new KV states with NEW SYS + NEW KB + Summaries.
    3. Sends completion signal webhook back to the main server.
    """
    client_id_str: str = payload.client_id or "default"
    log_message("pipeline", f"Starting summary & recompilation pipeline for client '{client_id_str}'...")

    state_files: List[Path] = list(STATES_DIR.glob("*.bin"))
    summaries: Dict[str, str] = {}

    # Step A: Summarize all active customer states
    for pfile in state_files:
        ckey = pfile.stem
        summ = extract_customer_summary(ckey, pfile)
        if summ:
            summaries[ckey] = summ

    # Step B: Re-compile new binary KV states for each customer
    compiled_count: int = 0
    for ckey, summ_text in summaries.items():
        success = prefill_and_save_kv_state(
            customer_key=ckey,
            system_prompt=payload.system_prompt,
            kb_content=payload.kb_content,
            persona=payload.persona,
            summary_text=summ_text
        )
        if success:
            compiled_count += 1

    # Step C: Send completion signal webhook back to main server
    signal_payload: Dict[str, Any] = {
        "status": "ready",
        "total_states_compiled": compiled_count
    }
    if payload.client_id:
        signal_payload["client_id"] = payload.client_id

    try:
        res = requests.post(payload.webhook_url, json=signal_payload, timeout=15)
        log_message("pipeline", f"Completion signal sent to webhook '{payload.webhook_url}'. HTTP {res.status_code}")
    except Exception as err:
        log_message("pipeline", f"Failed to send webhook completion signal to '{payload.webhook_url}': {err}")

tunnel_process: Optional[subprocess.Popen] = None

def start_cloudflare_tunnel() -> Optional[str]:
    """Launches cloudflared tunnel process to expose local port 8000."""
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
        for _ in range(15):
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

def update_github_dns(pat: str, org: str, public_url: str, repo_name: str) -> None:
    """
    Updates public tunnel URL in dynamic Cloudflare Worker DNS registry (key: 0bm/{repo_name} & {repo_name}),
    with direct GitHub API fallback if CF Worker is unreachable.
    """
    dns_keys: List[str] = [f"{SUPERKEY}/{repo_name}", repo_name]
    log_message("system", f"Updating dynamic DNS registry for repo '{repo_name}' with URL {public_url}...")
    
    # 1. Try Cloudflare Worker Endpoint
    cf_success: bool = False
    for dns_key in dns_keys:
        for attempt in range(1, 4):
            try:
                payload = {"key": dns_key, "value": public_url}
                res = requests.post("https://dns-manager.aakashmishra2050880.workers.dev/update", json=payload, timeout=10)
                if res.status_code == 200:
                    log_message("system", f"DNS updated successfully via CF Worker for key '{dns_key}' with URL {public_url}")
                    cf_success = True
                    break
                else:
                    log_message("system", f"CF Worker returned status code {res.status_code} for key '{dns_key}': {res.text}")
            except Exception as e:
                import random
                log_message("system", f"Error updating DNS via CF Worker (attempt {attempt}/3): {e}")
                time.sleep(random.uniform(1.0, 3.0))

    if cf_success:
        return

    # 2. Direct GitHub API Fallback (if CF Worker is down or rate-limited)
    if not pat:
        log_message("system", "No GITHUB_PAT provided. Skipping direct GitHub API DNS fallback.")
        return

    log_message("system", "Running direct GitHub API DNS fallback update...")
    try:
        from github import Github, Auth
        auth_obj: Auth.Token = Auth.Token(pat)
        g: Github = Github(auth=auth_obj)
        dns_repo = g.get_repo(f"{org}/dns")
        contents = dns_repo.get_contents("config.json")
        current_data: Dict[str, Any] = json.loads(contents.decoded_content.decode("utf-8")) if contents else {}
        
        if SUPERKEY not in current_data or not isinstance(current_data[SUPERKEY], dict):
            current_data[SUPERKEY] = {}
        current_data[SUPERKEY][repo_name] = public_url
        current_data[repo_name] = public_url
        
        new_content: str = json.dumps(current_data, indent=2)
        dns_repo.update_file(
            path="config.json",
            message=f"Direct DNS update for {repo_name} -> {public_url}",
            content=new_content,
            sha=contents.sha
        )
        log_message("system", f"Direct GitHub API DNS update succeeded for '{repo_name}'!")
    except Exception as gh_err:
        log_message("system", f"Direct GitHub API DNS update failed: {gh_err}")


async def drain_upstash_redis_queue() -> int:
    """
    Drains FIFO queued fallback messages from Upstash Redis per client key 'fallback_queue:{client_id}'
    on startup and processes them sequentially through the LLM pipeline.
    """
    redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
    redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
    client_id = os.getenv("CLIENT_ID", "default")
    queue_key = f"fallback_queue:{client_id}"

    if not redis_url or not redis_token:
        log_message("system", "[Redis Drain] Upstash Redis credentials not set. Skipping queue drain.")
        return 0

    log_message("system", f"[Redis Drain] Checking Upstash Redis queue '{queue_key}'...")
    drained_count = 0
    headers = {"Authorization": f"Bearer {redis_token}"}

    try:
        while True:
            # LPOP 1 element from FIFO queue
            pop_url = f"{redis_url}/lpop/{queue_key}"
            res = requests.get(pop_url, headers=headers, timeout=10)
            if res.status_code != 200:
                break
            
            data = res.json()
            payload_raw = data.get("result")
            if not payload_raw:
                break

            try:
                item = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
                user_msg = item.get("msg") or item.get("prompt") or item.get("userMessage") or ""
                p_num = item.get("phone_number") or item.get("cleanNumber") or ""

                if user_msg and p_num:
                    log_message("system", f"[Redis Drain] Processing queued message from {p_num}...")
                    await run_model_query(
                        prompt=user_msg,
                        phone_number=p_num
                    )
                    drained_count += 1
            except Exception as item_err:
                log_message("system", f"[Redis Drain] Error evaluating item: {item_err}")

        log_message("system", f"[Redis Drain] Finished draining {drained_count} queued messages from Redis.")
    except Exception as err:
        log_message("system", f"[Redis Drain] Error querying Upstash Redis: {err}")

    return drained_count

def get_current_repo_name() -> str:
    """
    Auto-detects repository name directly from git remote config or current working directory.
    Zero environment variables required!
    """
    try:
        out = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], text=True, stderr=subprocess.DEVNULL).strip()
        if out:
            repo = out.rstrip("/").removesuffix(".git").split("/")[-1]
            if repo:
                return repo
    except Exception:
        pass
    return Path(".").resolve().name

@asynccontextmanager
async def lifespan(app: FastAPI):
    pat: str = os.getenv("GITHUB_PAT", "") or os.getenv("GH_PAT", "")
    org: str = os.getenv("GITHUB_ORG", "LeadbaseAI-Official")
    repo_name: str = get_current_repo_name()

    duration_str: str = os.getenv("RUN_DURATION_HOURS", "5.5")
    try:
        duration_hours: float = float(duration_str)
    except ValueError:
        duration_hours = 5.5

    # 1. Download customer states & global seed from Hugging Face for this repo_name
    download_states_from_huggingface(repo_name)


    # 2. Warm up model weights
    log_message("system", "Warming up model weights...")
    try:
        get_llm()
        GLOBAL_CACHE_DIR.mkdir(exist_ok=True)
        log_message("system", "Model initialized successfully.")
    except Exception as warmup_err:
        log_message("system", f"Warning: model warmup failed: {warmup_err}")

    # 3. Drain Upstash Redis fallback queue
    await drain_upstash_redis_queue()

    # 4. Start 5:30h lifecycle timer
    threading.Thread(
        target=run_5_30_lifecycle_timer,
        args=(pat, org, repo_name, duration_hours),
        daemon=True
    ).start()

    # 5. Start Cloudflare Tunnel & update DNS
    public_url: Optional[str] = start_cloudflare_tunnel()
    if public_url:
        log_message("system", f"CLOUDFLARE TUNNEL ESTABLISHED SUCCESSFULLY! Address: {public_url}")
        update_github_dns(pat, org, public_url, repo_name)
    yield

app = FastAPI(title="Local GGUF LLM API Server", lifespan=lifespan)

@app.get("/health")
@app.get("/")
async def health_check() -> Dict[str, Any]:
    return {"status": "ok", "accepting": is_accepting_requests}

@app.post("/v1/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    if not is_accepting_requests:
        raise HTTPException(
            status_code=503,
            detail="Runner is currently recycling. Incoming message redirected to Redis fallback queue."
        )

    prompt_text = req.prompt or req.user_question
    if not prompt_text or not req.phone_number:
        raise HTTPException(status_code=400, detail="Parameters 'prompt' (or 'user_question') and 'phone_number' are required.")

    result = await run_model_query(
        prompt=prompt_text,
        phone_number=req.phone_number
    )
    if isinstance(result, str):
        raise HTTPException(status_code=500, detail=result)

    return {
        "response": result.get("response", ""),
        "phone_number": req.phone_number,
        "abandon_token": result.get("abandon_token")
    }


@app.post("/v1/state-update")

def receive_state_update(req: StateUpdateItem, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Receives compiled binary KV state from kv_worker, decompresses it,
    saves to GLOBAL_CACHE_DIR / "global.bin", triggers background KV rebuild,
    and dispatches ready signal to Agent-0 server.
    """
    try:
        c_id: str = req.client_id or "default"
        
        print("\n" + "═" * 65, flush=True)
        log_message("STATE_UPDATE", f"📥 RECEIVED COMPILED KV STATE for client_id='{c_id}'")
        log_message("STATE_UPDATE", f"   Base64 Payload Size: {len(req.state_bytes_base64)} chars")
        print("═" * 65, flush=True)
        
        compressed_bytes: bytes = base64.b64decode(req.state_bytes_base64)
        raw_bytes: bytes = gzip.decompress(compressed_bytes)
        payload_obj: Dict[str, Any] = pickle.loads(raw_bytes)
        
        state_data = payload_obj.get("state")
        tokens_data = payload_obj.get("tokens", [])
        
        log_message("STATE_UPDATE", f"Decompressed state size: {len(raw_bytes)} bytes (~{len(raw_bytes)/(1024*1024):.2f} MB), token count: {len(tokens_data)}")
        
        # Save to global cache directory
        GLOBAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        global_file: Path = GLOBAL_CACHE_DIR / "global.bin"
        with open(global_file, "wb") as gf:
            gf.write(raw_bytes)
        log_message("STATE_UPDATE", f"Saved global seed to disk: {global_file}")
        
        repo_name: str = get_current_repo_name()
        clear_huggingface_and_local_states(repo_name)


        # If rebuilt_customer_states provided (rebuild_states=True), save rebuilt states
        if req.rebuilt_customer_states:
            for p_num, c_b64 in req.rebuilt_customer_states.items():
                try:
                    c_bytes = gzip.decompress(base64.b64decode(c_b64))
                    c_obj = pickle.loads(c_bytes)
                    c_file = STATES_DIR / f"{p_num}.bin"
                    with open(c_file, "wb") as cf:
                        cf.write(c_bytes)
                    log_message("STATE_UPDATE", f"Saved rebuilt customer state for {p_num}")
                except Exception as r_err:
                    log_message("STATE_UPDATE", f"Error saving rebuilt state for {p_num}: {r_err}")

        # Trigger background KV cache rebuild for active conversations
        if state_data:
            queue_global_rebuild(state_data, tokens_data)
            log_message("STATE_UPDATE", f"✅ Successfully queued global KV cache rebuild for client '{c_id}'!")
            
        # Write kv_status=complete to Redis so wa-gateway drains the fallback queue
        background_tasks.add_task(send_kv_complete_to_redis, c_id)
            
        print("═" * 65 + "\n", flush=True)
        return {"status": "success", "client_id": c_id, "state_size_bytes": len(raw_bytes)}
    except Exception as ex:
        import traceback
        traceback.print_exc()
        log_message("STATE_UPDATE", f"❌ Failed to process received KV state update: {ex}")
        raise HTTPException(status_code=500, detail=f"State update failed: {str(ex)}")




@app.post("/v1/clear-chat")
async def clear_chat_state(req: ClearRequest) -> Dict[str, Any]:
    try:
        target_key = req.phone_number.strip()
        if target_key and target_key != "all":
            state_file = STATES_DIR / f"{target_key}.bin"
            if state_file.exists():
                state_file.unlink()
            if target_key in _ram_states_cache:
                del _ram_states_cache[target_key]
            log_message("system", f"Cleared conversation history cache for phone_number: {target_key}")
        else:
            for path in STATES_DIR.glob("*.bin"):
                path.unlink()
            _ram_states_cache.clear()
            log_message("system", "Cleared all customer conversation history caches")
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, access_log=False)
