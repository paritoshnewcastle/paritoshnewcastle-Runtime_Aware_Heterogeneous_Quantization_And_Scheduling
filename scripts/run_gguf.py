import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
paths = json.loads((repo_root / "config" / "paths.json").read_text())

llama_cli = Path(paths["llama_cpp_dir"]) / "llama-cli.exe"
model_path = (repo_root / paths["model_dir"] / "test_model.gguf").resolve()

cmd = [
    str(llama_cli),
    "-m", str(model_path),
    "-ngl", "33",
    "-c", "4096"
]

print(f"Running GGUF model: {model_path.name}")
print(f"Command: {' '.join(cmd)}")

try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print("\nExiting interactive chat...")
except subprocess.CalledProcessError as e:
    print(f"\nCommand failed with exit code {e.returncode}")
    sys.exit(e.returncode)
