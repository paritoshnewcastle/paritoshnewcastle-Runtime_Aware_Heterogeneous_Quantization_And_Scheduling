import subprocess
import json
import sys
from pathlib import Path

def probe_xrt_smi(xrt_smi_path="xrt-smi"):
    print(f"Probing {xrt_smi_path}...")
    
    import tempfile
    import os
    
    # 1. Platform Report
    print("\n--- Platform Report Schema ---")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_name = tmp.name
        
    res_platform = subprocess.run(
        [xrt_smi_path, "examine", "--report", "platform", "-f", "JSON", "-o", tmp_name, "--force"],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
    )
    if res_platform.returncode == 0:
        try:
            with open(tmp_name, 'r') as f:
                data = json.load(f)
            print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
    else:
        print(f"Command failed with return code {res_platform.returncode}")
        print(res_platform.stderr)
        print(res_platform.stdout)

    # 2. AIE Partitions Report
    print("\n--- AIE Partitions Report Schema ---")
    res_aie = subprocess.run(
        [xrt_smi_path, "examine", "--report", "aie-partitions", "-f", "JSON", "-o", tmp_name, "--force"],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
    )
    if res_aie.returncode == 0:
        try:
            with open(tmp_name, 'r') as f:
                data = json.load(f)
            print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
    else:
        print(f"Command failed with return code {res_aie.returncode}")
        print(res_aie.stderr)
        print(res_aie.stdout)
        
    try:
        os.remove(tmp_name)
    except:
        pass

if __name__ == "__main__":
    # Check paths.json for the path
    config_path = Path(__file__).resolve().parents[1] / "config" / "paths.json"
    xrt_path = "xrt-smi"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                paths = json.load(f)
                xrt_path = paths.get("xrt_smi_path", "xrt-smi")
        except Exception:
            pass
            
    probe_xrt_smi(xrt_path)
