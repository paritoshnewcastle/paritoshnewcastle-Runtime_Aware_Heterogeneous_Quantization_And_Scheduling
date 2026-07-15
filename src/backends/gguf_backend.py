import subprocess
import time
import threading
import re
from pathlib import Path
from .base_backend import BaseBackend, GenerationResult

class GGUFBackend(BaseBackend):
    def __init__(self, llama_cli_path: str, backend_type: str = "gpu", precision: str = "FP16"):
        self.llama_cli_path = llama_cli_path
        self.model_path = ""
        self.model_name = ""
        self.backend_type = backend_type
        self.precision = precision

    def load(self, model_path: str):
        self.model_path = model_path
        self.model_name = Path(model_path).name
        print(f"Loaded GGUF model path {self.model_path}")

    def generate(self, prompt: str, max_tokens: int) -> GenerationResult:
        if not self.model_path:
            raise RuntimeError("Model not loaded.")

        # Construct llama-cli command for generation
        cmd = [
            str(self.llama_cli_path),
            "-m", str(self.model_path),
            "-ngl", "33",
            "-c", "4096",
            "-p", prompt,
            "-n", str(max_tokens),
            "-st" # Single-turn conversation mode to prevent interactive prompt loop
        ]

        start_time = time.perf_counter()
        first_token_time = None
        generated_text = ""
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, bufsize=1)
        
        stderr_lines = []
        def drain_stderr():
            for line in process.stderr:
                stderr_lines.append(line)
                
        t = threading.Thread(target=drain_stderr, daemon=True)
        t.start()
        
        while True:
            char = process.stdout.read(1)
            if not char:
                break
            if first_token_time is None and char.strip():
                # Approximation: first non-empty output character means first token generated
                first_token_time = time.perf_counter()
            generated_text += char
            print(char, end='', flush=True)
            
        t.join()
        end_time = time.perf_counter()
        
        # Parse metrics. `llama-cli` usually prints summary metrics to stderr at the end
        stderr_output = "".join(stderr_lines)
        
        prompt_tokens = 0
        generated_tokens = 0
        
        # Robust parsing logic for llama.cpp stderr output
        for log_line in stderr_output.split('\n'):
            m_prompt = re.search(r'prompt eval time.*?/\s*(\d+)\s*tokens', log_line)
            if m_prompt:
                prompt_tokens = int(m_prompt.group(1))
            
            m_eval = re.search(r'^\s*llama_perf.*eval time.*?/\s*(\d+)\s*runs', log_line)
            if m_eval:
                generated_tokens = int(m_eval.group(1))

        # Extract actual generated text by stripping out UI clutter
        clean_text = generated_text
        prompt_marker = f"> {prompt}"
        if prompt_marker in clean_text:
            clean_text = clean_text.split(prompt_marker)[-1]
        
        if "[ Prompt:" in clean_text:
            clean_text = clean_text.split("[ Prompt:")[0]
            
        clean_text = clean_text.strip()
                
        # Parse TPS directly from the conversational UI output if available
        m_tps = re.search(r'Generation:\s*([\d.]+)\s*t/s', generated_text)
        if m_tps:
            tps_from_ui = float(m_tps.group(1))
        else:
            tps_from_ui = None

        # Fallback if stderr parsing fails
        if generated_tokens == 0:
            if tps_from_ui and (end_time - first_token_time) > 0:
                generated_tokens = int(tps_from_ui * (end_time - first_token_time))
            else:
                generated_tokens = max(1, len(clean_text.split())) # rough word count

        ttft = first_token_time - start_time if first_token_time else end_time - start_time
        decode_time = end_time - first_token_time if first_token_time else 0.0
        total_latency = end_time - start_time
        
        if tps_from_ui:
            tps = tps_from_ui
        else:
            tps = generated_tokens / decode_time if decode_time > 0 else 0.0

        return GenerationResult(
            generated_text=clean_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            ttft=ttft,
            decode_time=decode_time,
            total_latency=total_latency,
            decode_throughput_tps=tps,
            backend=self.backend_type,
            model_name=self.model_name,
            precision=self.precision
        )

    def unload(self):
        self.model_path = ""
