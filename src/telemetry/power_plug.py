import json
import threading
import paho.mqtt.client as mqtt

class PowerMonitor:
    def __init__(self, broker_host: str, broker_port: int, topic: str, power_json_path: str):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.power_json_path = power_json_path
        
        self._current_power_w = 0.0
        self._lock = threading.Lock()
        
        self.client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    clean_session=True
)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
    def _extract_power(self, payload: dict) -> float:
        try:
            keys = self.power_json_path.split(".")
            val = payload
            for k in keys:
                val = val[k]
            return float(val)
        except Exception:
            return 0.0

    def _on_connect(self, client, userdata, connect_flags, reason_code, properties):
        if not reason_code.is_failure:
            print(
                f"[PowerMonitor] Connected to MQTT broker. "
                f"Subscribing to {self.topic}"
            )
            client.subscribe(self.topic)
        else:
            print(f"[PowerMonitor] MQTT connection failed: {reason_code}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            power = self._extract_power(payload)
            with self._lock:
                self._current_power_w = power
        except Exception as e:
            # Silently ignore parsing errors to not spam logs
            pass

    def start(self):
        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=10)
            self.client.loop_start()
            return True
        except Exception as e:
            print(f"[PowerMonitor] Could not start MQTT loop: {e}")
            return False

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    @property
    def current_power_w(self) -> float:
        with self._lock:
            return self._current_power_w
