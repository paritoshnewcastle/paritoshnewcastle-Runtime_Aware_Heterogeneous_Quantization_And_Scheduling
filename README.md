/# Runtime-Aware Heterogeneous Quantization and Scheduling

## 1. Clarifying the Quantization Idea

**Do NOT initially think of quantization as:**

> “quantize the whole model.”

This is too simplistic and not very novel.

**Instead, consider:**

- **Runtime-aware heterogeneous quantization**
- Different accelerators prefer different precisions:

| Device | Best Precision |
| ------ | -------------- |
| CPU    | INT8 / FP16    |
| GPU    | FP16 / INT8    |
| NPU    | INT4 / INT8    |

**Implications:**

- Conversion overhead
- Activation transfer overhead
- Synchronization overhead
- Cache consistency issues

> These challenges create a novel optimization problem.

---

## 2. The Key Systems Problem

Example assignment:

| Layers | Device | Precision |
| ------ | ------ | --------- |
| 1–10   | GPU    | FP16      |
| 11–20  | NPU    | INT4      |
| 21–24  | CPU    | INT8      |

**Runtime challenges:**

- Activations move across devices
- Precision conversion occurs
- Synchronization happens
- Memory bandwidth stressed

**Consequences:**

- Increased latency
- Increased energy consumption
- Thermal instability

**Key insight:**

> The runtime now must optimize **both device selection and precision**, creating a stronger systems-oriented problem.

---

## 3. Avoid Overcomplication Initially

**Do NOT start with:**

- Arbitrary per-layer quantization
  > This becomes complex compiler/runtime research.

**Recommended practical approach:**

- Use **one quantized version per backend**:

| Backend | Model Version |
| ------- | ------------- |
| GPU     | FP16          |
| CPU     | INT8          |
| NPU     | INT4          |

**Runtime selects:**

- Execution backend
- Corresponding quantized model

> This avoids dynamic requantization, per-layer conversion, and excessive tensor conversions.

---

## 4. Better Paper Formulation

Instead of “dynamic quantization,” use:

**Backend-specialized multi-precision execution**

> Sounds stronger and more systems-oriented.

---

## 5. The Truly Novel Problem

Runtime jointly optimizes:

| Variable         | Decision          |
| ---------------- | ----------------- |
| Device           | CPU/GPU/NPU       |
| Precision        | FP16/INT8/INT4    |
| Thermal state    | Migrate or not    |
| Memory bandwidth | Avoid congestion  |
| Inference phase  | Prefill/Decode    |
| SLA target       | Latency or energy |

> This is **multi-objective runtime optimization**—SC-style work.

---

## 6. Decode Phase Changes Everything

**Phase characteristics:**

| Phase   | Hardware Preference                              |
| ------- | ------------------------------------------------ |
| Prefill | GPU-friendly, compute-heavy                      |
| Decode  | Memory-bound, token-by-token, often NPU-friendly |

**Example scheduling:**

| Phase             | Backend | Precision |
| ----------------- | ------- | --------- |
| Prefill           | GPU     | FP16      |
| Decode            | NPU     | INT4      |
| Thermal emergency | CPU     | INT8      |

---

## 7. Runtime Transfer Overhead

Three possible architectures:

**Option A — Whole-model migration (recommended initially)**

- Simple, low engineering effort
- Avoids layer synchronization
- Has migration overhead and duplicated caches

**Option B — Phase-level partitioning (best balance)**

- Prefill → GPU FP16, Decode → NPU INT4, Sampling → CPU
- Novel but manageable complexity
- Avoids per-layer synchronization

**Option C — Per-layer heterogeneous execution**

- Attention → GPU, MLP → NPU, KV → CPU
- Very novel but extremely hard
- Issues: activation transfer, synchronization, graph fragmentation, runtime stalls

> Start with **phase-level scheduling** for realism, novelty, and feasibility.

---

## 8. Strongest Architecture

```

```

            ┌──────────────────┐
            │ Runtime Scheduler │
            └────────┬─────────┘
                     │
     ┌───────────────┼────────────────┐
     │               │                │
     ▼               ▼                ▼

CPU INT8 GPU FP16 NPU INT4
│ │ │
└──── Shared KV Cache Manager ───┘

```

```

**Scheduler decisions depend on:**

- Temperature
- Bandwidth pressure
- Queue length
- Context length
- Decode rate

---

## 9. Overall Implementation Strategy

| Layer            | Goal                         |
| ---------------- | ---------------------------- |
| Infrastructure   | Inference + telemetry        |
| Characterization | Understand hardware behavior |
| Runtime          | Scheduling logic             |
| Evaluation       | Experiments + analysis       |

> Do not start with adaptive scheduling—characterization is key.

---

## Week-by-Week Implementation Plan

### Week 1 — Infrastructure + Hardware Characterization

**Goal:** Build a working heterogeneous inference pipeline and collect telemetry.

**Parts:**

1. **Environment Setup:**

   - Refer to official AMD docs

2. **Model Preparation:**
   - Small models: Llama-3.2-3B-Instruct Phi-3-mini, Qwen2.5-3B,
   - Create multiple precision variants per backend

3. **Benchmark Harness:**
   - Inputs: model, backend, prompt length, generation length
   - Outputs: TTFT, tokens/sec, latency, total runtime

4. **Telemetry Collection:**
   - CPU temp/power, GPU/NPU utilization, DRAM bandwidth, memory usage, frequency
   - Sample every 100ms

5. **Characterization Experiments:**
   - Short/long prompts, sustained inference
   - Measure thermal throttling, throughput collapse, bandwidth saturation

**Deliverables:**

- Working heterogeneous inference
- Telemetry system
- Latency, power, thermal, bandwidth data

---

### Week 2 — Phase-Aware Runtime

**Goal:** Separate prefill and decode phases, dispatch differently.

**Parts:**

1. **Understand SLM Phases**
2. **Implement Phase Detection**
3. **Static Phase Mapping**

| Phase    | Backend | Precision |
| -------- | ------- | --------- |
| Prefill  | GPU     | FP16      |
| Decode   | NPU     | INT4      |
| Sampling | CPU     | INT8      |

4. **KV Cache Handling:** Duplicate cache on all backends initially
5. **Measure Transfer Overheads:** Migration latency, backend switch cost

**Deliverables:** Heterogeneous phase-aware runtime with backend switching and overhead measurements.

---

### Week 3 — Thermal- and Bandwidth-Aware Adaptation

**Goal:** Add online adaptive scheduling.

**Parts:**

1. **Thermal Model:** Moving average, linear regression, EWMA
2. **Bandwidth Pressure Estimation:** Detect threshold exceedance
3. **Adaptive Scheduling Policy:** Decisions based on thermal/bandwidth state
4. **Scheduling Granularity:** Session-level and phase-level

**Deliverables:** Adaptive runtime with thermal- and bandwidth-aware migration.

---

### Week 4 — Evaluation and Stress Testing

**Parts:**

1. **Baselines:** CPU-only, GPU-only, NPU-only, Static heterogeneous, Adaptive runtime
2. **Workload Types:** Short/long prompts, multi-turn chat, batch inference, sustained workloads
3. **Metrics:** TTFT, tokens/sec, joules/token, throttling frequency, sustained throughput, migration overhead, bandwidth utilization
4. **Ablation Studies:** Disable thermal, bandwidth, and phase awareness to show contribution

---

### Week 5 — Refinement + Paper-Quality Analysis

**Parts:**

1. **Trace Visualization:** Thermal, throughput, migration, bandwidth
2. **Analyze Failure Cases:** Migration hurts, NPU saturation, cache overhead
3. **Improve Scheduler:** Hysteresis, cooldowns, predictive thresholds
4. **Reproducibility:** Scripts, configs, logging automation

---

## 10. Final Recommendation on Quantization

**Backend-specialized model variants:**

| Backend | Precision |
| ------- | --------- |
| GPU     | FP16      |
| CPU     | INT8      |
| NPU     | INT4      |

**Schedule by phase, adapt by thermal/bandwidth state, avoid per-layer migration complexity.**

> Realistic, novel, implementable, and publishable.

## Device details

# 1. Minisforum mini PC 

Device name	amd_ryzen_ai_pc
Processor	AMD Ryzen AI 9 HX 370 w/ Radeon 890M (2.00 GHz)
Installed RAM	32.0 GB (27.6 GB usable)
OS Windows 11 Pro

GPU AMD Radeon(TM) 890M Graphics
	Dedicated GPU memory	0.5/3.8 GB
	Shared GPU memory	0.2/13.8 GB
	GPU Memory	0.7/17.6 GB

NPU NPU Compute Accelerator Device
	Dedicated GPU memory	0.5/3.8 GB
	Shared GPU memory	3.4/13.8 GB
	GPU Memory	3.4/13.8 GB

# 2. Local Bytes Smart Plug
   - Wifi Power Monitoring Plug
   - Power Monitoring Preflashed and Preconfigured
   - Supports MQTT


# AMD Official Docs 
- https://ryzenai.docs.amd.com/en/latest/llm/overview.html => LLM Deployment overview
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/advanced/advancedrad/windows/llm/llamacpp.html => Rocm llamacpp
- https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html => OGA Flow 
