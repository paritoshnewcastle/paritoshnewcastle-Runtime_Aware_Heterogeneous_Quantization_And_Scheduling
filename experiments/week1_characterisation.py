import sys
import time
import json
import csv
import subprocess
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from src.telemetry.collector import TelemetryCollector
from src.backends.oga_backend import OGABackend
from src.backends.gguf_backend import GGUFBackend

def set_pmode_performance(xrt_smi_path: str):
    print("Setting NPU to Performance Mode...")
    try:
        subprocess.run([xrt_smi_path, "configure", "--pmode", "performance"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        print("Success.")
    except Exception as e:
        print(f"Failed to set performance mode: {e}")

# Provide dummy words for prompt generation
WORDS = "The quick brown fox jumps over the lazy dog and explores the vast mechanics of a beautiful spaceship orbiting the distant red planet. "

def generate_prompt(size: str) -> str:
    if size == "short": # ~50 words
        return WORDS * 3
    elif size == "medium": # ~500 words
        return WORDS * 30
    else: # "long" ~2000 words
        return WORDS * 125

def main():
    config_path = repo_root / "config" / "paths.json"
    paths = json.loads(config_path.read_text())
    
    xrt_path = paths.get("xrt_smi_path", "xrt-smi")
    set_pmode_performance(xrt_path)
    
    manifest_path = repo_root / "experiments" / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    BACKENDS = list(set([entry["backend"] for entry in manifest]))
    PROMPT_SIZES = ["short", "medium", "long"]
    REPETITIONS = 3
    
    results = []
    
    import random
    import uuid
    import datetime
    
    # Generate run_id for this benchmark suite execution
    suite_run_id = str(uuid.uuid4())
    
    # Generate all test conditions
    all_tests = []
    for entry in manifest:
        model_name = entry["model_name"]
        backend_type = entry["backend"]
        
        # Resolve artifact path relative to the model directory
        model_path = str(repo_root / paths["model_dir"] / entry["artifact"])
        
        for prompt_size in PROMPT_SIZES:
            for rep in range(1, REPETITIONS + 1):
                all_tests.append((model_name, backend_type, model_path, prompt_size, rep))
                    
    # Randomise the order of all tests to prevent systemic thermal carry-over
    random.shuffle(all_tests)
    
    current_backend_type = None
    current_model_name = None
    backend = None

    for idx, (model_name, backend_type, model_path, prompt_size, rep) in enumerate(all_tests):
        print(f"\n===========================================")
        print(f"Test {idx+1}/{len(all_tests)}: Model: {model_name} | Backend: {backend_type.upper()} | Prompt: {prompt_size} | Rep: {rep}")
        
        # Load backend if it changed
        if backend_type != current_backend_type or model_name != current_model_name:
            if backend:
                backend.unload()
                backend = None
                
            if not Path(model_path).exists():
                print(f"Skipping {backend_type} (Model path missing or invalid: {model_path})")
                continue
                
            try:
                if backend_type in ["npu", "hybrid"]:
                    backend = OGABackend(backend_type=backend_type, precision="INT4")
                else:
                    llama_cli = str(Path(paths["llama_cpp_dir"]) / "llama-cli.exe")
                    backend = GGUFBackend(llama_cli_path=llama_cli, backend_type="gpu", precision="FP16")
                
                backend.load(model_path)
                print(f"  -> Warm-up run for {backend_type}...")
                backend.generate("Warm-up prompt", 10)
                warmup_completed = True
                
                current_backend_type = backend_type
                current_model_name = model_name
            except Exception as e:
                print(f"Failed to load backend {backend_type} or during warmup: {e}")
                current_backend_type = None
                current_model_name = None
                continue
        else:
            warmup_completed = False

        # Active Cooldown
        idle_cooldown_s = 15
        print(f"  -> Cooldown for {idle_cooldown_s} seconds to dissipate thermal carry-over...")
        time.sleep(idle_cooldown_s)

        prompt_text = generate_prompt(prompt_size)
        max_tokens = 100
        
        collector = TelemetryCollector(str(config_path))
        collector.start()
        
        # Collect idle baseline before generation
        time.sleep(2) 
        
        collector.set_active_backend(backend_type, f"sweep_{prompt_size}_{rep}")
        collector.set_inference_phase("prefill")
        
        started_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        try:
            res = backend.generate(prompt_text, max_tokens)
            collector.set_inference_phase("idle")
            collector.update_tokens_decoded(res.generated_tokens, res.decode_throughput_tps)
        except Exception as e:
            print(f"    Generation failed: {e}")
            res = None
            
        time.sleep(2) # Collect cooldown
        collector.stop()
        
        if res and res.generated_tokens > 0:
            # Process telemetry
            inf_samples = [s for s in collector.samples if s.inference_phase != "idle"]
            baseline_samples = [s for s in collector.samples if s.inference_phase == "idle" and s.timestamp < inf_samples[0].timestamp] if inf_samples else []
            
            valid_power_samples = len(inf_samples)
            
            if baseline_samples:
                start_cpu_temp_c = baseline_samples[-1].cpu_tdie
                start_gpu_temp_c = baseline_samples[-1].gpu_temp
            else:
                start_cpu_temp_c = 0
                start_gpu_temp_c = 0
            
            if inf_samples:
                max_cpu_temp = max([s.cpu_tdie for s in inf_samples])
                max_gpu_temp = max([s.gpu_temp for s in inf_samples])
                avg_sys_power = sum([s.system_power_w for s in inf_samples]) / len(inf_samples)
                max_dram = max([s.dram_used_mb for s in inf_samples])
                min_dram = min([s.dram_used_mb for s in inf_samples])
            else:
                max_cpu_temp, max_gpu_temp, avg_sys_power = 0, 0, 0
                max_dram, min_dram = 0, 0
                
            delta_cpu_temp_c = max_cpu_temp - start_cpu_temp_c
            delta_gpu_temp_c = max_gpu_temp - start_gpu_temp_c
                
            decode_time = res.decode_time
            # Energy = Power (W) * Time (s) = Joules
            sys_energy_j = avg_sys_power * decode_time
            sys_energy_per_token = (sys_energy_j / res.generated_tokens) if avg_sys_power > 0 else 0.0
                
            results.append({
                "run_id": suite_run_id,
                "started_at_utc": started_at_utc,
                "run_order": idx + 1,
                "model": model_name,
                "backend": backend_type,
                "prompt_size": prompt_size,
                "rep": rep,
                "ambient_temp_c": "N/A",
                "start_cpu_temp_c": start_cpu_temp_c,
                "start_gpu_temp_c": start_gpu_temp_c,
                "idle_cooldown_s": idle_cooldown_s,
                "warmup_completed": warmup_completed,
                "valid_power_samples": valid_power_samples,
                "ttft_s": res.ttft,
                "tps": res.decode_throughput_tps,
                "max_cpu_temp": max_cpu_temp,
                "max_gpu_temp": max_gpu_temp,
                "delta_cpu_temp_c": delta_cpu_temp_c,
                "delta_gpu_temp_c": delta_gpu_temp_c,
                "dram_delta_mb": max_dram - min_dram,
                "sys_energy_per_token_j": sys_energy_per_token,
                "total_latency_s": res.total_latency
            })
            
            print(f"    TPS: {res.decode_throughput_tps:.2f} | TTFT: {res.ttft:.2f}s | CPU Peak: {max_cpu_temp}°C (Delta: {delta_cpu_temp_c:.1f}°C) | DRAM Allocation Delta: {max_dram - min_dram:.1f}MB")
            
    if backend:
        backend.unload()
            
    # Save Results
    out_csv = repo_root / "results" / "characterisation_week1.csv"
    out_csv.parent.mkdir(exist_ok=True)
    
    if results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        
        print(f"\n===========================================")
        print(f"Characterisation Sweep Complete! Saved data to {out_csv}")
        print(f"===========================================\n")
        
        # Print Summary Table
        print(f"{'Backend':<10} | {'Prompt':<8} | {'Avg TTFT':<10} | {'Avg TPS':<8} | {'Peak CPU°C':<10} | {'Sys mJ/Tok':<10}")
        print("-" * 75)
        
        for b in BACKENDS:
            for p in PROMPT_SIZES:
                subset = [r for r in results if r["backend"] == b and r["prompt_size"] == p]
                if not subset:
                    continue
                avg_ttft = sum([r["ttft_s"] for r in subset]) / len(subset)
                avg_tps = sum([r["tps"] for r in subset]) / len(subset)
                peak_cpu = max([r["max_cpu_temp"] for r in subset])
                avg_sys_j = sum([r["sys_energy_per_token_j"] for r in subset]) / len(subset)
                
                print(f"{b:<10} | {p:<8} | {avg_ttft:<10.2f} | {avg_tps:<8.2f} | {peak_cpu:<10.1f} | {avg_sys_j*1000:<10.2f}")
                
if __name__ == "__main__":
    main()
