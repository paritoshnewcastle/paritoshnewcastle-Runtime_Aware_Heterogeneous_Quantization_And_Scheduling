import sys
from pathlib import Path
import json

# Add src to python path so we can import backends
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from src.backends.oga_backend import OGABackend
from src.backends.gguf_backend import GGUFBackend

def main():
    paths = json.loads((repo_root / "config" / "paths.json").read_text())
    prompt = "Tell me a very short joke."
    max_tokens = 50

    print("=== Testing GGUF Backend (GPU) ===")
    gguf_model_path = str(repo_root / paths["model_dir"] / "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    llama_cli = str(Path(paths["llama_cpp_dir"]) / "llama-cli.exe")
    
    try:
        gguf_backend = GGUFBackend(llama_cli_path=llama_cli, backend_type="gpu", precision="FP16")
        gguf_backend.load(gguf_model_path)
        result = gguf_backend.generate(prompt, max_tokens)
        print(f"GGUF Generation Result:\n{result}\n")
    except Exception as e:
        print(f"GGUF Backend failed: {e}\n")

    print("=== Testing OGA Backend (NPU) ===")
    npu_model_path = str(repo_root / paths["model_dir"] / "Llama-3.2-3B-Instruct_rai_1.7.1_npu_16K")
    
    try:
        npu_backend = OGABackend(backend_type="npu", precision="INT4")
        npu_backend.load(npu_model_path)
        result = npu_backend.generate(prompt, max_tokens)
        print(f"OGA Generation Result:\n{result}\n")
    except Exception as e:
        print(f"OGA Backend failed: {e}\n")

    print("=== Testing OGA Backend (Hybrid) ===")
    hybrid_model_path = str(repo_root / paths["model_dir"] / "Llama-3.2-3B-Instruct_rai_1.7.1_hybrid")
    
    try:
        hybrid_backend = OGABackend(backend_type="hybrid", precision="INT4")
        hybrid_backend.load(hybrid_model_path)
        result = hybrid_backend.generate(prompt, max_tokens)
        print(f"OGA Generation Result:\n{result}\n")
    except Exception as e:
        print(f"OGA Backend failed: {e}\n")

if __name__ == "__main__":
    main()
