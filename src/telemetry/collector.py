import time
import json
import threading
import subprocess
import csv
import os
from pathlib import Path
from dataclasses import dataclass, fields, asdict
import psutil

# Optional dependencies
try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False

from .power_plug import PowerMonitor

@dataclass
class TelemetrySample:
    # Time
    timestamp: float               # time.perf_counter() relative to experiment start

    # CPU (via PyLibreHardwareMonitor or fallback)
    cpu_package_temp: float        # °C — package/die temperature
    cpu_tdie: float                # °C — Tdie (AMD-specific, more accurate than Tctl)
    cpu_util_pct: float            # % — from psutil

    # GPU (via PyLibreHardwareMonitor)
    gpu_temp: float                # °C
    gpu_util_pct: float            # %
    gpu_mem_used_mb: float         # MB — shared VRAM usage

    # NPU (via xrt-smi JSON, cached at 1Hz)
    npu_active: bool               # True if any HW context is Active
    npu_submitted_ops: int         # Cumulative submitted op count (delta = utilisation proxy)
    npu_completed_ops: int
    npu_power_w: float             # Watts — from xrt-smi platform report
    npu_perf_mode: str             # "Default" | "Performance" | "Powersaver"
    npu_migrations: int
    npu_errors: int
    npu_gops: float
    npu_egops: float
    npu_fps: float
    npu_latency_us: float
    npu_total_mem_mb: float

    # Memory / Bandwidth
    dram_used_mb: float            # MB — psutil virtual_memory
    dram_pressure_mb: float        # MB — memory used delta (proxy for bandwidth pressure)

    # Power
    system_power_w: float          # Watts — smart plug MQTT

    # Inference context (set by benchmark harness, not hardware)
    inference_phase: str           # "prefill" | "decode" | "idle"
    tokens_decoded: int            # Count so far in this run
    decode_rate_tps: float         # Rolling 5-token tokens/sec

    # Scheduling metadata (populated in Week 3, default empty string/0)
    active_backend: str            # "npu" | "hybrid" | "gpu" | "cpu"
    migration_event: bool          # True on the sample where backend switched
    scheduler_reason: str          # "" | "thermal" | "bandwidth" | "phase_transition"

class TelemetryCollector:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, "r") as f:
            self.paths = json.load(f)

        self.samples = []
        self._running = False
        
        # State variables set by harness
        self._inference_phase = "idle"
        self._tokens_decoded = 0
        self._decode_rate_tps = 0.0
        self._active_backend = ""
        self._migration_event = False
        self._scheduler_reason = ""
        self._state_lock = threading.Lock()
        
        self.start_time = 0.0
        self._stop_event = threading.Event()
        
        # DRAM tracking
        self._prev_dram_used_mb = psutil.virtual_memory().used / (1024 * 1024)

        # 1. Init NPU (xrt-smi)
        self.xrt_smi_path = self.paths.get("xrt_smi_path", "xrt-smi")
        self._xrt_available = self._probe_xrt_smi()
        self._npu_cache = {
            "active": False,
            "submitted_ops": 0,
            "completed_ops": 0,
            "power_w": 0.0,
            "perf_mode": "",
            "migrations": 0,
            "errors": 0,
            "gops": 0.0,
            "egops": 0.0,
            "fps": 0.0,
            "latency_us": 0.0,
            "total_mem_mb": 0.0,
        }
        self._npu_lock = threading.Lock()
        
        # 2. Init Thermal Sensors
        self.thermal_available = False
        self.thermal_source = ""
        self._lhm = None
        self._wmi = None
        self._init_lhm()
        
        # 3. Init Power Plug (MQTT)
        self.power_monitor = None
        mqtt_cfg = self.paths.get("mqtt")
        if mqtt_cfg:
            self.power_monitor = PowerMonitor(
                broker_host=mqtt_cfg.get("broker_host", "127.0.0.1"),
                broker_port=mqtt_cfg.get("broker_port", 1883),
                topic=mqtt_cfg.get("topic", "tele/smartplug/SENSOR"),
                power_json_path=mqtt_cfg.get("power_json_path", "ENERGY.Power")
            )
            started = self.power_monitor.start()
            if not started:
                print("[Telemetry] MQTT PowerMonitor failed to start.")
                self.power_monitor = None
        else:
            print("[Telemetry] No 'mqtt' config found in paths.json. Skipping smart plug telemetry.")

        self._collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._xrt_thread = threading.Thread(target=self._xrt_poll_loop, daemon=True)
        if self.thermal_source == "WMI_FALLBACK":
            self._wmi_thermal_cache = {"cpu_temp": -1.0}
            self._wmi_lock = threading.Lock()
            self._wmi_thread = threading.Thread(target=self._wmi_poll_loop, daemon=True)

    def _probe_xrt_smi(self) -> bool:
        try:
            result = subprocess.run(
                [self.xrt_smi_path, "examine"],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            if result.returncode == 0:
                return True
            else:
                print(f"[Telemetry] WARNING: xrt-smi probe failed with return code {result.returncode}")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"[Telemetry] WARNING: xrt-smi not found or timed out.")
            return False

    def _init_lhm(self) -> bool:
        try:
            from HardwareMonitor.Hardware import Computer, IComputer
            self._lhm = Computer()
            self._lhm.IsCpuEnabled = True
            self._lhm.IsGpuEnabled = True
            self._lhm.Open()
            self.thermal_available = True
            self.thermal_source = "LHM"
            print("[Telemetry] LibreHardwareMonitor initialised via HardwareMonitor package.")
            
            # Debug prints to discover exact hardware type strings on AMD APU
            for hw in self._lhm.Hardware:
                hw.Update()
                print(f"  Hardware: {hw.Name} | Type: {hw.HardwareType}")
                for s in hw.Sensors:
                    print(f"    Sensor: {s.Name} | Type: {s.SensorType} | Value: {s.Value}")
            
            return True
        except Exception as e:
            print(f"[Telemetry] WARNING: HardwareMonitor failed: {e}")
            print(f"[Telemetry] Are you running as Administrator?")
            print(f"[Telemetry] Falling back to WMI thermal zone (slow, inaccurate)...")
            
            if WMI_AVAILABLE:
                self.thermal_available = True
                self.thermal_source = "WMI_FALLBACK"
                # Keep LHM open flag false, wmi will be used
            else:
                print(f"[Telemetry] wmi package not installed. Cannot use fallback.")
                self.thermal_available = False
                self.thermal_source = "NONE"
            return False

    def _wmi_poll_loop(self):
        # Only runs if in Tier 2
        import pythoncom
        pythoncom.CoInitialize()
        c = wmi.WMI(namespace="root\\wmi")
        while self._running:
            try:
                # Temperature is in tenths of degrees Kelvin
                zones = c.MSAcpi_ThermalZoneTemperature()
                if zones:
                    temp_k = zones[0].CurrentTemperature / 10.0
                    temp_c = temp_k - 273.15
                    with self._wmi_lock:
                        self._wmi_thermal_cache["cpu_temp"] = temp_c
            except Exception as e:
                pass
            if self._stop_event.wait(timeout=1.0):
                break
        pythoncom.CoUninitialize()

    def _xrt_poll_loop(self):
        if not self._xrt_available:
            return
            
        import tempfile
        import os
        my_pid = str(os.getpid())
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_name = tmp.name

        while self._running:
            try:
                # 1. Platform power report
                power_w = 0.0
                perf_mode = ""
                res_platform = subprocess.run(
                    [self.xrt_smi_path, "examine", "--report", "platform", "-f", "JSON", "-o", tmp_name, "--force"],
                    capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW
                )
                if res_platform.returncode == 0:
                    try:
                        with open(tmp_name, 'r') as f:
                            data = json.load(f)
                        dev = data.get("devices", [{}])[0]
                        platform = dev.get("platforms", [{}])[0]
                        
                        raw_power = platform.get("electrical", {}).get("power_consumption_watts", "N/A")
                        power_w = 0.0 if raw_power == "N/A" else float(raw_power)
                        perf_mode = platform.get("status", {}).get("power_mode", "")
                    except Exception:
                        pass

                # 2. AIE partitions report
                npu_active = False
                npu_submitted = 0
                npu_completed = 0
                npu_migrations = 0
                npu_errors = 0
                npu_gops = 0.0
                npu_egops = 0.0
                npu_fps = 0.0
                npu_latency_us = 0.0
                npu_total_mem_mb = 0.0
                
                res_aie = subprocess.run(
                    [self.xrt_smi_path, "examine", "--report", "aie-partitions", "-f", "JSON", "-o", tmp_name, "--force"],
                    capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW
                )
                if res_aie.returncode == 0:
                    try:
                        with open(tmp_name, 'r') as f:
                            data = json.load(f)
                        dev = data.get("devices", [{}])[0]
                        aie = dev.get("aie_partitions", {})
                        
                        raw_mem = aie.get("total_memory_usage", "0 MB")
                        npu_total_mem_mb = float(raw_mem.split()[0]) if raw_mem != "N/A" else 0.0
                        
                        for partition in aie.get("partitions", []):
                            for ctx in partition.get("hw_contexts", []):
                                is_ours = ctx.get("pid") == my_pid
                                
                                if is_ours and ctx.get("status") == "Active":
                                    npu_active = True
                                
                                if is_ours:
                                    def _f(v): return 0.0 if v == "N/A" else float(v)
                                    def _i(v): return 0 if v == "N/A" else int(v)
                                    
                                    npu_submitted  += _i(ctx.get("command_submissions", "0"))
                                    npu_completed  += _i(ctx.get("command_completions", "0"))
                                    npu_migrations += _i(ctx.get("migrations", "0"))
                                    npu_errors     += _i(ctx.get("errors", "0"))
                                    npu_gops       += _f(ctx.get("gops", "N/A"))
                                    npu_egops      += _f(ctx.get("egops", "N/A"))
                                    npu_fps        += _f(ctx.get("fps", "N/A"))
                                    npu_latency_us  = _f(ctx.get("latency", "N/A"))
                    except Exception:
                        pass

                with self._npu_lock:
                    self._npu_cache.update({
                        "active":         npu_active,
                        "submitted_ops":  npu_submitted,
                        "completed_ops":  npu_completed,
                        "power_w":        power_w,
                        "perf_mode":      perf_mode,
                        "migrations":     npu_migrations,
                        "errors":         npu_errors,
                        "gops":           npu_gops,
                        "egops":          npu_egops,
                        "fps":            npu_fps,
                        "latency_us":     npu_latency_us,
                        "total_mem_mb":   npu_total_mem_mb,
                    })
                    
            except Exception as e:
                print(f"[Telemetry] NPU thread error: {e}")
            
            if self._stop_event.wait(timeout=1.0):
                break
                
        try:
            os.remove(tmp_name)
        except:
            pass

    def _get_all_lhm_sensors(self) -> dict:
        result = {
            "cpu_package_temp": -1.0, "cpu_tdie": -1.0,
            "gpu_temp": -1.0, "gpu_util_pct": 0.0, "gpu_mem_used_mb": 0.0
        }
        if not self._lhm:
            return result
        for hw in self._lhm.Hardware:
            hw.Update()
            hw_type = str(hw.HardwareType)
            for sensor in hw.Sensors:
                name = str(sensor.Name).lower()
                stype = str(sensor.SensorType)
                val = float(sensor.Value) if sensor.Value is not None else -1.0
                if hw_type == "Cpu":
                    if stype == "Temperature" and "tctl/tdie" in name:
                        result["cpu_tdie"] = val
                        result["cpu_package_temp"] = val # Use Tdie as package temp for APU
                elif hw_type == "GpuAmd":
                    if stype == "Temperature" and "vr soc" in name:
                        result["gpu_temp"] = val
                    elif stype == "Load" and "core" in name:
                        result["gpu_util_pct"] = val
                    elif stype == "SmallData" and "gpu memory used" in name:
                        result["gpu_mem_used_mb"] = val
        return result

    def _collect_loop(self):
        while self._running:
            start_ts = time.perf_counter()
            rel_ts = start_ts - self.start_time
            
            # --- CPU Metrics ---
            cpu_util = psutil.cpu_percent()
            
            if self.thermal_source == "LHM":
                lhm = self._get_all_lhm_sensors()
                cpu_pkg_temp = lhm["cpu_package_temp"]
                cpu_tdie = lhm["cpu_tdie"]
                gpu_temp = lhm["gpu_temp"]
                gpu_util = lhm["gpu_util_pct"]
                gpu_mem = lhm["gpu_mem_used_mb"]
            elif self.thermal_source == "WMI_FALLBACK":
                with self._wmi_lock:
                    cpu_pkg_temp = self._wmi_thermal_cache["cpu_temp"]
                cpu_tdie = cpu_pkg_temp
                gpu_temp = -1.0
                gpu_util = 0.0
                gpu_mem = 0.0
            else:
                cpu_pkg_temp = -1.0
                cpu_tdie = -1.0
                gpu_temp = -1.0
                gpu_util = 0.0
                gpu_mem = 0.0

            # --- NPU Metrics ---
            with self._npu_lock:
                npu_act = self._npu_cache["active"]
                npu_sub = self._npu_cache["submitted_ops"]
                npu_comp = self._npu_cache["completed_ops"]
                npu_pwr = self._npu_cache["power_w"]
                npu_mode = self._npu_cache["perf_mode"]
                npu_mig = self._npu_cache["migrations"]
                npu_err = self._npu_cache["errors"]
                npu_gops = self._npu_cache["gops"]
                npu_egops = self._npu_cache["egops"]
                npu_fps = self._npu_cache["fps"]
                npu_lat = self._npu_cache["latency_us"]
                npu_mem = self._npu_cache["total_mem_mb"]
                
            # --- Power Metrics ---
            sys_power = self.power_monitor.current_power_w if self.power_monitor else 0.0
            
            # --- Memory Metrics ---
            curr_dram_mb = psutil.virtual_memory().used / (1024 * 1024)
            dram_pressure = curr_dram_mb - self._prev_dram_used_mb
            self._prev_dram_used_mb = curr_dram_mb
            
            # --- Harness Metadata ---
            with self._state_lock:
                phase = self._inference_phase
                tok_dec = self._tokens_decoded
                tps = self._decode_rate_tps
                backend = self._active_backend
                mig = self._migration_event
                rsn = self._scheduler_reason
                
                # Clear migration event flag after recording it
                if self._migration_event:
                    self._migration_event = False

            sample = TelemetrySample(
                timestamp=rel_ts,
                cpu_package_temp=cpu_pkg_temp,
                cpu_tdie=cpu_tdie,
                cpu_util_pct=cpu_util,
                gpu_temp=gpu_temp,
                gpu_util_pct=gpu_util,
                gpu_mem_used_mb=gpu_mem,
                npu_active=npu_act,
                npu_submitted_ops=npu_sub,
                npu_completed_ops=npu_comp,
                npu_power_w=npu_pwr,
                npu_perf_mode=npu_mode,
                npu_migrations=npu_mig,
                npu_errors=npu_err,
                npu_gops=npu_gops,
                npu_egops=npu_egops,
                npu_fps=npu_fps,
                npu_latency_us=npu_lat,
                npu_total_mem_mb=npu_mem,
                dram_used_mb=curr_dram_mb,
                dram_pressure_mb=dram_pressure,
                system_power_w=sys_power,
                inference_phase=phase,
                tokens_decoded=tok_dec,
                decode_rate_tps=tps,
                active_backend=backend,
                migration_event=mig,
                scheduler_reason=rsn
            )
            
            self.samples.append(sample)
            
            if len(self.samples) % 600 == 0 and len(self.samples) > 0:
                self._flush_checkpoint()
                
            elapsed = time.perf_counter() - start_ts
            sleep_time = max(0, 0.1 - elapsed)
            time.sleep(sleep_time)

    def _flush_checkpoint(self):
        checkpoint_dir = Path("results")
        checkpoint_dir.mkdir(exist_ok=True)
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        chk_path = checkpoint_dir / f"checkpoint_{ts_str}.csv"
        self.save_to_csv(str(chk_path))
        print(f"[Telemetry] Saved checkpoint to {chk_path}")

    # --- Public API ---
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self.start_time = time.perf_counter()
        self._collect_thread.start()
        self._xrt_thread.start()
        if hasattr(self, '_wmi_thread'):
            self._wmi_thread.start()
        print("[Telemetry] Collection started.")

    def stop(self):
        self._running = False
        self._stop_event.set()
        self._collect_thread.join()
        self._xrt_thread.join()
        if hasattr(self, '_wmi_thread'):
            self._wmi_thread.join()
        if self.power_monitor:
            self.power_monitor.stop()
        if self._lhm:
            self._lhm.Close()
        print(f"[Telemetry] Collection stopped. Total samples: {len(self.samples)}")

    def save_to_csv(self, filepath: str):
        if not self.samples:
            return
            
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(TelemetrySample)])
            writer.writeheader()
            for s in self.samples:
                writer.writerow(asdict(s))
                
    def save_metadata(self, filepath: str, metadata: dict):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    def set_inference_phase(self, phase: str):
        with self._state_lock:
            self._inference_phase = phase
            
    def update_tokens_decoded(self, count: int, rate_tps: float = 0.0):
        with self._state_lock:
            self._tokens_decoded = count
            self._decode_rate_tps = rate_tps
            
    def set_active_backend(self, backend: str, reason: str = ""):
        with self._state_lock:
            if self._active_backend != "" and self._active_backend != backend:
                self._migration_event = True
            self._active_backend = backend
            self._scheduler_reason = reason
