# 6G Intent-to-Service-Profile Framework (TMP 3.0)

A knowledge-guided LLM framework that converts natural-language 6G service
intents into validated, orchestration-ready service profiles for integrated
Terrestrial/Non-Terrestrial (TN/NTN) network deployments.

---

## How it works

```
Natural-language intent
        │
        ▼
┌───────────────────┐
│  S1 · Classify    │  Which of the 4 service classes? (TC / MC / RC / MaC)
├───────────────────┤
│  S2 · Extract     │  Pull slot values directly from the intent text (LLM)
├───────────────────┤
│  S3 · Clarify     │  Ask only what is missing or conflicting (KG-driven)
├───────────────────┤
│  S5 · Assemble    │  Build the structured service profile (LLM)
├───────────────────┤
│  S6 · Deploy mode │  Assign TN/NTN deployment mode from the knowledge base
└───────────────────┘
        │
        ▼
  Validated JSON service profile
```

**Service classes**

| ID  | Full name             | Primary objective                          |
|-----|-----------------------|--------------------------------------------|
| TC  | TimeCritical          | Sub-ms latency, industrial/vehicular control |
| MC  | MissionCritical       | Resilience-first, public safety, fallback  |
| RC  | RemoteConnectivity    | NTN-primary, maritime/rural/aerial reach   |
| MaC | MassiveConnectivity   | Millions of low-power IoT devices          |

---

## Requirements

- Python 3.10 or later
- A GPU with at least 6 GB VRAM (recommended) or CPU-only (slower)
- The model file: `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`

---

## Installation

**1. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

For GPU acceleration (CUDA):

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

**3. Place the model file**

Create a `models/` folder inside the export directory and put the GGUF file there:

```
export/
└── models/
    └── Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

You can download the model from Hugging Face:

```
https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF
```

Or if the model is already available on your machine, pass its path with `--model`.

---

## Running the interactive demo

```bash
python run_interactive.py
```

With a custom model path:

```bash
python run_interactive.py --model /path/to/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

Save the output JSON to a file:

```bash
python run_interactive.py --output my_profile.json
```

Print stage-level debug messages:

```bash
python run_interactive.py --verbose
```

---

## Running the batch evaluation

Runs the full pipeline over the 43-sample labeled dataset and writes two output
files to `eval/`:

```bash
python run_eval.py
```

Filter by service class, request type, or specific IDs:

```bash
python run_eval.py --class TC                    # only TimeCritical samples
python run_eval.py --type contradictory          # only contradictory samples
python run_eval.py --ids TC-01 MC-03             # specific samples
python run_eval.py --verbose                     # print dialogue turns
python run_eval.py --compare                     # diff vs previous results
```

**Output files**

| File | Description |
|---|---|
| `eval/transcripts.json` | Per-sample dialogue turns + all M1–M8 metric values |
| `eval/results_full.json` | Aggregate statistics + per-sample metric summary |

---

## Generating new dataset samples

Uses the LLM to generate paraphrases, ambiguous variants, or contradictory
variants from the existing seed intents:

```bash
python generate_dataset.py --dry-run             # preview without LLM calls
python generate_dataset.py                       # all seeds, 2 variants each
python generate_dataset.py --class TC --type paraphrased --n 3
python generate_dataset.py --ids TC-01 MC-02 --type contradictory
```

**Output files** (written to `eval/`)

| File | Description |
|---|---|
| `eval/generated_samples.py` | Python source, appendable to `dataset.py` after review |
| `eval/generated_samples.json` | Same samples in JSON for easy inspection |

> **Note:** Fields marked `# REVIEW` in the Python output (`expected_slots`,
> `expected_conflicts`) need manual verification before the samples are added
> to the labeled dataset. The LLM does not guarantee correct ground-truth labels.

---

## Example session

```
  Your intent: We need real-time remote surgery support over a satellite link
               for a hospital ship in the Pacific.

──────────────────────────────────────────────────────────────────
  Stage 1 · Intent Classification
──────────────────────────────────────────────────────────────────
  Detected class:  MissionCritical

──────────────────────────────────────────────────────────────────
  Stage 2 · Slot Extraction from Intent Text
──────────────────────────────────────────────────────────────────
  ✓  vertical_domain = Healthcare  [extracted]
  ✓  geographic_scope = Maritime   [extracted]
  ✓  deployment_scenario = Maritime [extracted]

  Still needed: ['isolation_level', 'priority_level', 'cost_profile',
                 'fallback_mode', 'ntn_tier']

──────────────────────────────────────────────────────────────────
  Stage 3 · Clarification Dialogue
──────────────────────────────────────────────────────────────────

  ?  What level of network isolation do you need?
     Options: Dedicated, Shared, BestEffort
  > Dedicated

  ?  What is the service priority level?
     Options: Critical, High, Medium, Low
  > Critical

  ...

──────────────────────────────────────────────────────────────────
  Output · Orchestration-Ready Service Profile
──────────────────────────────────────────────────────────────────
{
  "service_class": "MissionCritical",
  "vertical_domain": "Healthcare",
  "kpi_profile": { "latency_ms": 10, "reliability": 0.99999, ... },
  "traffic_pattern": "BurstyRealtime",
  "geographic_scope": "Maritime",
  "isolation_level": "Dedicated",
  "priority_level": "Critical",
  "cost_profile": "Premium",
  "deployment_mode": "NTN-only",
  "class_slots": { ... }
}
```

---

## Directory structure

```
export/
├── run_interactive.py      Interactive demo — real user dialogue in the terminal
├── run_eval.py             Batch evaluation — runs all 43 labeled samples
├── generate_dataset.py     Dataset generator — LLM-assisted sample creation
├── requirements.txt        Python dependencies
├── README.md               This file
│
├── models/                 Put your GGUF model file here
│   └── Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
│
├── eval/
│   ├── dataset.py          43 labeled EvalSample definitions (seed through contradictory)
│   ├── metrics.py          Metric computation: M1–M8, SampleResult, AggregateMetrics
│   ├── transcripts.json    (generated) Per-sample dialogue + metrics
│   └── results_full.json   (generated) Aggregate + per-sample summary
│
├── pipeline/
│   ├── models.py           Data structures: ServiceIntent, DialogueState, SlotValue, ...
│   ├── llm_client.py       LLM wrapper: classify_intent, extract_slots, parse_answer
│   └── dialogue_runner.py  Pipeline orchestrator: Stages 1–6
│
└── ontology/
    ├── types.py             KPI types and slot definition dataclasses
    ├── ontology_v3.py       Service-class ontology: slots, constraints, conflict rules
    └── kg.py                Knowledge graph: conflict checking, termination, deployment mode
```

---

## Output format

The framework outputs a JSON object with the following top-level fields:

| Field | Description |
|---|---|
| `service_class` | Resolved class: `TimeCritical`, `MissionCritical`, `RemoteConnectivity`, or `MassiveConnectivity` |
| `vertical_domain` | Industry vertical (e.g. `Healthcare`, `Energy`, `Maritime`) |
| `kpi_profile` | KPI envelope: latency, reliability, availability, throughput, density |
| `traffic_pattern` | Traffic model (e.g. `DeterministicPeriodic`, `BurstyRealtime`) |
| `geographic_scope` | Coverage scope (e.g. `Campus`, `Urban`, `Maritime`) |
| `isolation_level` | `Dedicated`, `Shared`, or `BestEffort` |
| `priority_level` | `Critical`, `High`, `Medium`, or `Low` |
| `cost_profile` | `Economy`, `Standard`, or `Premium` |
| `deployment_mode` | TN/NTN mode assigned by the knowledge base: `TN-only`, `NTN-only`, or `Hybrid` |
| `class_slots` | All class-specific elicited parameters (cycle time, fallback mode, NTN tier, etc.) |

---

## Troubleshooting

**`llama_cpp` not found**
Make sure you installed inside the virtualenv:
```bash
pip install llama-cpp-python
```

**Model loads but inference is very slow**
The model is running on CPU. Install the CUDA build:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

**`Model file not found` error**
Pass the full path explicitly:
```bash
python run_interactive.py --model /full/path/to/model.gguf
```

**Classification returns wrong class**
Try rephrasing the intent with more specific technical keywords
(e.g. "latency-critical" for TC, "fallback required" for MC,
"satellite" for RC, "smart meters" for MaC).

