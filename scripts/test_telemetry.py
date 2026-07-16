import sys
import time
import json
import csv
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from src.telemetry.collector import TelemetryCollector
from src.backends.gguf_backend import GGUFBackend

def main():
    print("=== Telemetry Collection Test ===")
    config_path = repo_root / "config" / "paths.json"
    
    print("1. Initializing telemetry collector...")
    collector = TelemetryCollector(str(config_path))
    
    print("2. Starting telemetry collection (Running for a few seconds before inference)...")
    collector.start()
    time.sleep(3) # Collect idle baseline
    
    print("3. Running GGUF Backend (GPU) inference to generate load...")
    paths = json.loads(config_path.read_text())
    
    gguf_model_path = str(repo_root / paths["model_dir"] / "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    llama_cli = str(Path(paths["llama_cpp_dir"]) / "llama-cli.exe")
    prompt = "Explain the mechanics of a spaceship in a detailed paragraph."
    max_tokens = 100
    
    try:
        collector.set_active_backend("gpu", "test_run")
        collector.set_inference_phase("prefill")
        
        gguf_backend = GGUFBackend(llama_cli_path=llama_cli, backend_type="gpu", precision="FP16")
        gguf_backend.load(gguf_model_path)
        
        print("Generating...")
        result = gguf_backend.generate(prompt, max_tokens)
        
        collector.set_inference_phase("idle")
        collector.update_tokens_decoded(result.generated_tokens, result.decode_throughput_tps)
        
        print(f"Generation complete.")
        print(f"Tokens: {result.generated_tokens}, TPS: {result.decode_throughput_tps:.2f}, TTFT: {result.ttft:.2f}s")
        print("Waiting 2 seconds to collect cool-down metrics...")
        time.sleep(2)
        
    except Exception as e:
        print(f"Inference failed (but telemetry should still be collected): {e}")
        time.sleep(2)
        
    print("4. Stopping telemetry collector...")
    collector.stop()
    
    out_csv = repo_root / "results" / "test_telemetry_gpu.csv"
    collector.save_to_csv(str(out_csv))
    print(f"Saved telemetry to {out_csv}")
    
    if out_csv.exists():
        with open(out_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"\nSuccessfully collected {len(rows)} samples.")
            
            if rows:
                print("\nSample [0] (Idle):")
                print(json.dumps(rows[0], indent=2))
                
                mid_index = len(rows) // 2
                print(f"\nSample [{mid_index}] (Mid-Inference):")
                print(json.dumps(rows[mid_index], indent=2))

if __name__ == "__main__":
    main()
