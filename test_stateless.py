import time
import os
from pathlib import Path
from llama_cpp import Llama

def run_benchmark() -> None:
    # Find the GGUF model file
    model_path = Path("Qwen3.5-0.8B-Q4_K_M.gguf")
    if not model_path.exists():
        # Look in model/ subfolder as well
        model_path = Path("model/Qwen3.5-0.8B-Q4_K_M.gguf")
        
    if not model_path.exists():
        # Fallback to scanning root for any GGUF
        ggufs = list(Path(".").glob("*.gguf"))
        if ggufs:
            model_path = ggufs[0]
        else:
            print("[Error] No GGUF model file found. Please ensure you have downloaded Qwen3.5-0.8B-Q4_K_M.gguf.")
            return

    print(f"[1/5] Loading model weights from: {model_path} ...")
    t0 = time.time()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_threads=2,
        flash_attn=True
    )
    print(f"Model loaded in {time.time() - t0:.2f} seconds.")

    # Generate ~2000 tokens of dummy prompt data
    dummy_word = "hello "
    prompt = dummy_word * 2000
    
    print("[2/5] Tokenizing dummy prompt (~2000 tokens) ...")
    tokens = llm.tokenize(prompt.encode("utf-8"))
    token_count = len(tokens)
    print(f"Total prompt tokens: {token_count}")

    # 1. Prefill / Process the 2000 tokens
    print(f"[3/5] Prefilling KV Cache with {token_count} tokens ...")
    t_prefill_start = time.time()
    # Evaluate prompt tokens
    llm.eval(tokens)
    t_prefill_end = time.time()
    prefill_duration = t_prefill_end - t_prefill_start
    print(f"Prefill complete in {prefill_duration:.2f} seconds ({token_count / prefill_duration:.2f} tokens/sec).")

    # 2. Save state using llama.cpp state API
    print("[4/5] Saving context state to SSD buffer ...")
    t_save_start = time.time()
    state_obj = llm.save_state()
    t_save_end = time.time()
    print(f"State saved in {t_save_end - t_save_start:.4f} seconds.")

    # Save to disk as a benchmark check
    import pickle
    state_file = Path("test_convo.bin")
    with open(state_file, "wb") as f:
        pickle.dump(state_obj, f)

    # Reset context to simulate clear stateless slot
    llm.reset()

    # 3. Restore state from SSD
    print("[5/5] Restoring context state from SSD binary ...")
    t_load_start = time.time()
    with open(state_file, "rb") as f:
        loaded_state_obj = pickle.load(f)
    llm.load_state(loaded_state_obj)
    print(f"State restored in {time.time() - t_load_start:.4f} seconds.")

    # 4. Generate next 200 tokens
    print("\n[Benchmark] Generating 200 tokens ...")
    t_gen_start = time.time()
    
    # We feed the model one token at a time to generate the next 200 tokens
    generated_tokens = []
    last_token = tokens[-1]
    
    for i in range(200):
        # Evaluate last token and get next token prediction
        llm.eval([last_token])
        logits = llm._scores
        next_token = logits.argmax()
        generated_tokens.append(next_token)
        last_token = next_token
        
    t_gen_end = time.time()
    gen_duration = t_gen_end - t_gen_start
    print(f"\nGeneration complete in {gen_duration:.2f} seconds ({200 / gen_duration:.2f} tokens/sec).")

    # Cleanup benchmark file
    if state_file.exists():
        os.remove(state_file)

if __name__ == "__main__":
    run_benchmark()
