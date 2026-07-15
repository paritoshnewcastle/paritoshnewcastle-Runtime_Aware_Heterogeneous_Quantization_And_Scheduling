from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class GenerationResult:
    # Text
    generated_text: str
    prompt_tokens: int          # Number of input tokens (for prefill analysis)
    generated_tokens: int       # Number of output tokens generated

    # Timing
    ttft: float                 # Seconds from submission -> first output token (= prefill latency)
    decode_time: float          # Seconds from first token -> last token
    total_latency: float        # ttft + decode_time (wall clock end-to-end)

    # Throughput
    decode_throughput_tps: float  # generated_tokens / decode_time

    # Backend metadata
    backend: str                # "npu" | "hybrid" | "gpu"
    model_name: str             # e.g. "Llama-3.2-3B"
    precision: str              # "INT4" | "INT8" | "FP16"

class BaseBackend(ABC):
    @abstractmethod
    def load(self, model_path: str):
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> GenerationResult:
        pass

    @abstractmethod
    def unload(self):
        pass
