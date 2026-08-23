import os
import time
import threading
import requests
from model.engine import log_message, _eval_lock
from model.cache_manager import upload_states_to_huggingface

is_accepting_requests: bool = True

def trigger_self_workflow_dispatch(pat: str, org: str, repo_name: str) -> bool:
    """Dispatches GitHub Actions workflow trigger for self repository default branch."""
    if not pat or repo_name == "test":
        log_message("lifecycle", "[Self Dispatch] Skipping self workflow dispatch (test mode or no PAT).")
        return False

    url = f"https://api.github.com/repos/{org}/{repo_name}/actions/workflows/workflow.yml/dispatches"
    headers = {
        "Authorization": f"token {pat}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "LeadBaseAI-Runner"
    }
    payload = {"ref": "main"}
    log_message("lifecycle", f"[Self Dispatch] Triggering new runner dispatch on {org}/{repo_name}...")
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        if res.status_code == 204:
            log_message("lifecycle", "[Self Dispatch] Successfully dispatched new runner workflow!")
            return True
        else:
            log_message("lifecycle", f"[Self Dispatch] Workflow dispatch returned status code {res.status_code}: {res.text}")
    except Exception as err:
        log_message("lifecycle", f"[Self Dispatch] Error dispatching workflow: {err}")
    return False

def run_5_30_lifecycle_timer(pat: str, org: str, repo_name: str, duration_hours: float = 5.5) -> None:
    """
    Background timer reaching 5:30h mark, stopping incoming request acceptance,
    uploading states to Hugging Face, triggering new runner dispatch, and exiting cleanly.
    """
    global is_accepting_requests
    duration_seconds: float = duration_hours * 3600
    sync_lead_time = 15 * 60  # 15 minutes before duration limit
    sleep_first = max(0.0, duration_seconds - sync_lead_time)

    log_message("lifecycle", f"5:30 Uptime timer started. Handover scheduled in {sleep_first / 60:.1f} minutes.")
    time.sleep(sleep_first)

    log_message("lifecycle", "5:30 Uptime mark reached. Stopping request acceptance (HTTP 503 fallback redirect)...")
    is_accepting_requests = False


    # Wait 3 seconds for active in-flight query evaluation locks to clear
    time.sleep(3)
    with _eval_lock:
        log_message("lifecycle", "All in-flight queries completed. Starting instant Hugging Face upload...")

    # Upload all states to Hugging Face under repo_name
    upload_success = upload_states_to_huggingface(repo_name)
    log_message("lifecycle", f"Hugging Face state upload finished for '{repo_name}' (success={upload_success}).")

    # Trigger new runner workflow dispatch
    trigger_self_workflow_dispatch(pat, org, repo_name)

    log_message("lifecycle", "Handover complete! Exiting runner process immediately.")
    os._exit(0)


