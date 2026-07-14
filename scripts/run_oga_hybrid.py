import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
paths = json.loads((repo_root / "config" / "paths.json").read_text())

model_chat = Path(paths["oga_example_dir"]) / "model_chat.py"
model_dir = (repo_root / paths["model_dir"] / "Llama-3.2-3B-Instruct_rai_1.7.1_hybrid").resolve()

cmd = [sys.executable, str(model_chat), "-m", str(model_dir)]

subprocess.run(cmd, cwd=paths["oga_example_dir"], check=True)
