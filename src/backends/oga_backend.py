import onnxruntime_genai as og
import time
from pathlib import Path
from .base_backend import BaseBackend, GenerationResult

class OGABackend(BaseBackend):
    def __init__(self, backend_type: str, precision: str):
        self.model = None
        self.tokenizer = None
        self.backend_type = backend_type # "npu" or "hybrid"
        self.precision = precision
        self.model_name = ""

    def load(self, model_path: str):
        print(f"Loading OGA model from {model_path}...")
        self.model = og.Model(model_path)
        self.tokenizer = og.Tokenizer(self.model)
        self.model_name = Path(model_path).name

    def generate(self, prompt: str, max_tokens: int) -> GenerationResult:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded.")

        tokens = self.tokenizer.encode(prompt)
        prompt_tokens_count = len(tokens)
        
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=prompt_tokens_count + max_tokens)

        generator = og.Generator(self.model, params)
        generator.append_tokens(tokens)
        
        start_time = time.perf_counter()
        first_token_time = None
        generated_tokens_ids = []

        while not generator.is_done():
            generator.generate_next_token()
            
            if first_token_time is None:
                first_token_time = time.perf_counter() # Prefill complete
            
            new_token = generator.get_next_tokens()[0]
            generated_tokens_ids.append(new_token)

        end_time = time.perf_counter()
        
        # Calculate metrics
        ttft = first_token_time - start_time if first_token_time else 0.0
        decode_time = end_time - first_token_time if first_token_time else 0.0
        total_latency = end_time - start_time
        num_generated = len(generated_tokens_ids)
        tps = num_generated / decode_time if decode_time > 0 else 0.0

        # TODO: Use TokenizerStream for streaming decoding to handle multi-byte tokens properly
        generated_text = self.tokenizer.decode(generated_tokens_ids)
        
        del generator # Explicitly delete generator to avoid memory leaks

        return GenerationResult(
            generated_text=generated_text,
            prompt_tokens=prompt_tokens_count,
            generated_tokens=num_generated,
            ttft=ttft,
            decode_time=decode_time,
            total_latency=total_latency,
            decode_throughput_tps=tps,
            backend=self.backend_type,
            model_name=self.model_name,
            precision=self.precision
        )

    def unload(self):
        self.model = None
        self.tokenizer = None
