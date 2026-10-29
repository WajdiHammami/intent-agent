# Intent Agent — Knowledge-Guided LLM for Intent-Driven Service Orchestration

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Llama--3--8B%20(llama.cpp)-000000)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20prototype-blue)

Turn a messy, plain-English service request into a **validated, machine-ready
configuration** — without trusting the language model to get the hard decisions
right.

An LLM is great at *reading* an informal request ("we need real-time control for
60 welding robots on one factory floor, premium budget"). It is unreliable at
*deciding* whether that request is complete, self-consistent, and policy-valid.
This project keeps the LLM on language tasks and hands every decision to a small,
auditable **knowledge graph**. The payoff, measured on a labeled benchmark:
**+20 points of policy validity and +35 points of contradiction detection versus
an LLM-only pipeline, while asking 3.8× fewer clarifying questions.**

It's the reference implementation behind a research paper on 6G network
orchestration, but the pattern — *LLM for understanding, symbolic layer for
decisions* — applies to any domain where an agent's output has to be correct, not
just plausible: form intake, config generation, compliance checks, structured
data extraction.

---

## Why this is worth a look

- **A hybrid architecture that actually earns its keep.** The split between LLM
  and knowledge graph isn't decorative — the ablation study quantifies exactly
  what the symbolic layer buys you (see Results).
- **Deterministic where it counts.** Completeness checks, conflict detection, and
  termination are pure-Python rules over a typed graph. The same input always
  produces the same decision, and every decision is traceable to a rule.
- **A real evaluation harness, not a demo.** 8 metrics, a 72-sample labeled
  dataset, per-class and per-request-type breakdowns, a component ablation, and a
  "how much does knowledge coverage matter" sweep — all runnable from the CLI.
- **Runs on a single consumer GPU.** 4-bit quantized Llama-3-8B via llama.cpp,
  ~6 GB VRAM. A swappable Claude-API backend is included for higher accuracy.
- **Clean, dependency-light code.** One LLM wrapper, one graph module, one
  orchestrator. No framework lock-in, no hidden state.

**Skills on display:** LLM application engineering · retrieval/knowledge-graph
reasoning · prompt and output-schema design · evaluation & benchmark design ·
ablation methodology · clean Python architecture · reproducible research tooling.

---

## How it works

```
Natural-language request
        │
        ▼
┌─────────────────────┐
│  S1 · Classify      │   LLM — pick 1 of 4 service classes (a soft guess the
│                     │         graph can later revise or escalate)
├─────────────────────┤
│  S2 · Extract       │   LLM — pull whatever slot values the text already gives
├─────────────────────┤
│  S3 · Clarify loop  │   GRAPH — ask only for what's missing, conflicting, or
│                     │           class-ambiguous; the LLM just parses answers
├─────────────────────┤
│  S5 · Assemble      │   Build the structured profile
├─────────────────────┤
│  S6 · Deploy mode   │   GRAPH — assign TN-only / NTN-only / Hybrid from context
└─────────────────────┘
        │
        ▼
  Validated JSON profile
```

| Does the *language* | Does the *thinking* |
|---|---|
| Classify intent, extract fields from free text, parse typed values from user answers | Which fields are mandatory, which combinations are contradictory, when to stop asking, which deployment mode fits |
| `pipeline/llm_client.py` | `ontology/` (the knowledge graph) |

The LLM runs at temperature 0 and is asked for JSON; a tolerant parser recovers
from formatting drift. Because the **graph** — not the model — rules on
completeness and consistency, conflict detection is deterministic and the final
profile always satisfies the service ontology.

<details>
<summary><b>Concrete example of graph-driven conflict detection</b></summary>

> *"Sub-millisecond V2X platooning for highway trucks — and use the cheapest 5G
> plan available."*

The LLM extracts `priority=Critical` and `cost_profile=LowCost`. The graph holds
a `conflict_rule(Critical, LowCost)`, so instead of silently emitting an
impossible profile, the pipeline flags the contradiction and asks the user to
resolve it. An LLM-only pipeline, asked to self-check, misses this most of the
time — which is exactly what the Pure-LLM ablation below measures.

</details>

---

## Results

Full pipeline over the 72-sample labeled benchmark (18 per class),
Llama-3-8B-Instruct Q4_K_M, **zero-shot — no fine-tuning**:

| Metric | Overall | TC | MC | RC | MaC |
|---|:--:|:--:|:--:|:--:|:--:|
| Classification accuracy      | **90.3%** | 100% | 94.4% | 100% | 66.7% |
| Policy validity rate         | **95.8%** | 83.3% | 100% | 100% | 100% |
| Avg. clarification turns     | **0.57**  | 0.83 | 0.17 | 0.50 | 0.78 |
| Profile completeness         | 99.3% | | | | |
| Contradiction detection acc. | 83.3% | | | | |

**The knowledge graph vs. an LLM-only baseline** — same classifier and extractor,
but the graph's reasoning replaced by the LLM deciding completeness and conflicts
from its own knowledge (40-sample contradictory subset):

| | Policy validity | Contradiction detection | Clarifying questions |
|---|:--:|:--:|:--:|
| LLM-only            | baseline     | baseline      | baseline |
| **+ Knowledge graph** | **+20 pts** | **+35 pts**   | **3.8× fewer** |

The graph doesn't help the model *classify* better — it makes the downstream
decisions **reliable and cheap**. That's the whole thesis, and it's measured, not
asserted.

Reproduce: `python run_eval.py` and `python run_pure_llm.py --contradictory-only`.

---

## Tech stack

| | |
|---|---|
| **Language** | Python 3.10+ (standard library + dataclasses; no heavy framework) |
| **Inference** | [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) running Llama-3-8B-Instruct (Q4_K_M, 4-bit) locally; optional Anthropic Claude backend |
| **Knowledge layer** | Hand-authored typed graph — required/conditional slots, conflict rules, class escalation, deployment-mode recommendations |
| **Evaluation** | Custom harness: 8 metrics, LLM-simulated user for multi-turn dialogue, ablation + coverage-sweep runners |

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place the model under models/ (git-ignored), then:
python run_interactive.py
```

Model: `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` from
[bartowski/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF),
or point any script at an existing file with `--model /path/to/model.gguf`.
For CUDA: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall`.

**Interactive demo**

```bash
python run_interactive.py --verbose          # show each stage's decisions
```

**Reproduce the experiments**

```bash
python run_eval.py                            # main 72-sample evaluation
python run_pure_llm.py --contradictory-only   # knowledge graph vs. LLM-only
python run_ablation.py                        # turn components on/off
python run_kg_sweep.py --fractions 0 25 50 75 100   # does graph coverage matter?
```

**Grow the dataset** (LLM proposes variants; labels are flagged for human review)

```bash
python generate_dataset.py --class TC --type paraphrased --n 3
```

---

## Dataset & evaluation design

`eval/dataset.py` is one self-contained module — no external data files:

| Collection | Size | Purpose |
|---|:--:|---|
| `ALL_SAMPLES` | 72 | Main set — 18 per class (9 seed, 3 paraphrased, 3 ambiguous, 3 contradictory) |
| `CONTRADICTORY_SUBSET` | 40 | Every conflict-rule type — drives the ablation & coverage sweep |
| `NON_CONTRADICTORY` | 60 | The well-formed samples inside `ALL_SAMPLES` |

Every sample carries a ground-truth class, an expected slot subset, and (for
contradictory samples) the conflict that *should* be caught. **Labels are
hand-authored and verified against the ontology — never taken from LLM output.**
The LLM stages run zero-shot, so there's no train/test split to worry about.
Multi-turn dialogue is evaluated with an LLM standing in as the user, driven by a
per-sample scenario script.

---

## Project structure

```
.
├── run_interactive.py     Interactive terminal demo
├── run_eval.py            Main 72-sample evaluation
├── run_pure_llm.py        Ablation: knowledge graph replaced by the LLM
├── run_ablation.py        Component ablation (extraction / conflict checking)
├── run_kg_sweep.py        Conflict-rule coverage sweep
├── generate_dataset.py    LLM-assisted, human-reviewed dataset expansion
│
├── pipeline/
│   ├── models.py          ServiceIntent, DialogueState, SlotValue, …
│   ├── llm_client.py      LLM wrapper — local llama.cpp or Claude API
│   └── dialogue_runner.py Stage orchestration + clarification dialogue
│
├── ontology/
│   ├── types.py           Graph edge-type dataclasses, numeric KPI profiles
│   ├── ontology_v3.py     The four service classes, scenarios, deploy modes
│   └── kg.py              Typed graph queries: conflicts, termination, mode
│
└── eval/
    ├── dataset.py         Labeled samples + the three collections above
    └── metrics.py         The 8 metrics and aggregation
```

Four service classes model the span of 6G traffic: **TimeCritical** (sub-ms
industrial control), **MissionCritical** (resilient public-safety with fallback),
**RemoteConnectivity** (satellite/rural reach), and **MassiveConnectivity**
(millions of low-power IoT devices).

---

## About

Built by **Wajdi Hammami** ([@WajdiHammami](https://github.com/WajdiHammami)),
Polytechnique Montreal, as the reference implementation for the paper *"Knowledge-
Guided LLM for Intent-Driven Service Orchestration in Integrated Terrestrial and
Non-Terrestrial 6G Networks"* (Hammami, Azam, Cherkaoui).

Open to related work — LLM application engineering, agentic pipelines, and
retrieval/knowledge-graph systems. Released under the [MIT License](LICENSE).
