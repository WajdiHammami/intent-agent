#!/usr/bin/env python3
"""
generate_dataset.py — LLM-Assisted Dataset Generator

Generates new EvalSample entries from the existing seed samples by asking the
LLM to produce paraphrases, ambiguous variants, or contradictory variants.

Output
------
  eval/generated_samples.py   — Python source with new EvalSample definitions
                                 (ready to append to dataset.py)
  eval/generated_samples.json — Same samples in JSON format (easier to inspect)

Usage
-----
    python generate_dataset.py                          # all seed samples, 2 per seed
    python generate_dataset.py --class TC               # only TimeCritical seeds
    python generate_dataset.py --type paraphrased       # paraphrases only
    python generate_dataset.py --type contradictory     # contradictory only
    python generate_dataset.py --n 3                    # 3 variants per seed
    python generate_dataset.py --ids TC-01 MC-02        # specific seeds only
    python generate_dataset.py --output my_samples.json # custom output path

The generated samples need a human review pass before being added to the
labeled dataset — the LLM does not guarantee correct expected_slots or
expected_conflicts.  Fields that require human verification are marked with
a "# REVIEW" comment in the Python output.

Generation modes
----------------
  paraphrased   — rephrase the intent while preserving all semantics
  ambiguous     — introduce genuine ambiguity (vague scope, mixed terminology)
  contradictory — insert two mutually exclusive requirements in one intent
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval.dataset import ALL_SAMPLES, EvalSample
from pipeline.llm_client import LLMClient

EVAL_DIR = Path(__file__).parent / "eval"
OUT_PY   = EVAL_DIR / "generated_samples.py"
OUT_JSON = EVAL_DIR / "generated_samples.json"

# Seed samples only (generation source)
SEED_SAMPLES = [s for s in ALL_SAMPLES if s.request_type == "seed"]


# ── Prompt builders ────────────────────────────────────────────────────────────

_PARAPHRASE_SYSTEM = (
    "You are a technical writer specialising in 5G/6G network service requirements. "
    "Given a service intent, produce a natural-language paraphrase that preserves "
    "ALL technical semantics (service class, QoS requirements, deployment context) "
    "but uses different phrasing, vocabulary, or sentence structure. "
    "Do NOT change any numeric values. Do NOT add or remove requirements. "
    "Respond with a JSON object only — no prose, no markdown fences."
)

_AMBIGUOUS_SYSTEM = (
    "You are a technical writer specialising in 5G/6G network service requirements. "
    "Given a service intent, produce an ambiguous variant where ONE requirement is "
    "genuinely unclear: use vague terminology, omit a key qualifier, or blend two "
    "scenarios so the service class is not immediately obvious. "
    "Keep all other details intact. "
    "Respond with a JSON object only — no prose, no markdown fences."
)

_CONTRADICTORY_SYSTEM = (
    "You are a technical writer specialising in 5G/6G network service requirements. "
    "Given a service intent, produce a contradictory variant that contains two "
    "mutually exclusive requirements (e.g., 'economy cost' AND 'dedicated slice', "
    "or 'best-effort' AND 'sub-millisecond latency'). "
    "The contradiction must be explicit and detectable from the text alone. "
    "State the contradiction clearly in the 'conflict_description' field. "
    "Respond with a JSON object only — no prose, no markdown fences."
)


def _build_paraphrase_prompt(sample: EvalSample) -> str:
    return (
        f'Original intent: "{sample.intent}"\n\n'
        'Return JSON:\n'
        '{\n'
        '  "intent": "<paraphrased intent text>",\n'
        '  "scenario": "<matching deployment scenario for the LLM user simulator>"\n'
        '}'
    )


def _build_ambiguous_prompt(sample: EvalSample) -> str:
    return (
        f'Original intent: "{sample.intent}"\n\n'
        'Return JSON:\n'
        '{\n'
        '  "intent": "<ambiguous intent text>",\n'
        '  "scenario": "<matching deployment scenario for the LLM user simulator>",\n'
        '  "ambiguity_description": "<one sentence: what is ambiguous and why>"\n'
        '}'
    )


def _build_contradictory_prompt(sample: EvalSample) -> str:
    return (
        f'Original intent: "{sample.intent}"\n\n'
        'Return JSON:\n'
        '{\n'
        '  "intent": "<intent text with an embedded contradiction>",\n'
        '  "scenario": "<matching deployment scenario for the LLM user simulator>",\n'
        '  "conflict_description": "<one sentence: which two requirements conflict>"\n'
        '}'
    )


# ── LLM generation ─────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract JSON dict from a response that may contain markdown fences."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON found in LLM response:\n{text[:300]}")


def generate_variant(
    llm: LLMClient,
    sample: EvalSample,
    variant_type: str,
) -> dict | None:
    """
    Ask the LLM to generate one variant of the given seed sample.

    Returns a dict with keys: intent, scenario, (optionally) conflict_description
    or ambiguity_description. Returns None on parse failure.
    """
    model = llm._get_model()

    if variant_type == "paraphrased":
        system = _PARAPHRASE_SYSTEM
        user   = _build_paraphrase_prompt(sample)
    elif variant_type == "ambiguous":
        system = _AMBIGUOUS_SYSTEM
        user   = _build_ambiguous_prompt(sample)
    elif variant_type == "contradictory":
        system = _CONTRADICTORY_SYSTEM
        user   = _build_contradictory_prompt(sample)
    else:
        raise ValueError(f"Unknown variant type: {variant_type}")

    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=512,
        temperature=0.7,   # some diversity for generation (vs 0.0 for eval)
    )
    text = response["choices"][0]["message"]["content"].strip()

    try:
        return _extract_json(text)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"    [parse error] {exc}")
        return None


# ── Sample assembly ────────────────────────────────────────────────────────────

def _next_id(base_id: str, variant_type: str, index: int, existing_ids: set[str]) -> str:
    """Generate a new sample ID that doesn't clash with existing ones."""
    prefix_map = {
        "paraphrased":   "PAR",
        "ambiguous":     "AMB",
        "contradictory": "CF",
    }
    prefix = prefix_map.get(variant_type, "GEN")
    # Try to preserve the class prefix: TC-01 → TC-PAR-01
    parts = base_id.split("-")
    class_prefix = parts[0] if len(parts) >= 2 else "XX"
    base = f"{class_prefix}-{prefix}-{index:02d}"
    candidate = base
    n = 1
    while candidate in existing_ids:
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def build_eval_sample(
    base_sample: EvalSample,
    variant: dict,
    variant_type: str,
    new_id: str,
) -> EvalSample:
    """Assemble an EvalSample from the LLM-generated variant dict."""
    expected_conflicts: list[str] = []
    if variant_type == "contradictory":
        desc = variant.get("conflict_description", "")
        if desc:
            expected_conflicts = [desc]

    return EvalSample(
        id=new_id,
        intent=variant["intent"],
        scenario=variant.get("scenario", base_sample.scenario),
        expected_class=base_sample.expected_class,
        # expected_slots intentionally left empty — requires human review
        expected_slots={},
        expected_conflicts=expected_conflicts,
        request_type=variant_type,
        scenario_label=base_sample.scenario_label,
        gold_deployment_mode=base_sample.gold_deployment_mode,
    )


# ── Output formatting ──────────────────────────────────────────────────────────

def _py_str(s: str) -> str:
    """Format a potentially multi-line string for Python source output."""
    if "\n" not in s and len(s) < 80:
        return f'"{s}"'
    # Use parenthesised string concatenation for long strings
    lines = textwrap.wrap(s, width=72)
    parts = "\n            ".join(f'"{line} "' for line in lines)
    return f"(\n            {parts}\n        )"


def to_python_source(samples: list[EvalSample], base_ids: dict[str, str]) -> str:
    """Render a list of EvalSamples as Python source (appendable to dataset.py)."""
    lines = [
        '"""',
        'Generated samples — AUTO-GENERATED by generate_dataset.py',
        '',
        'Fields marked with  # REVIEW  need human verification before use.',
        '"""',
        '',
        'from eval.dataset import EvalSample',
        '',
        'GENERATED_SAMPLES = [',
    ]
    for s in samples:
        src_id = base_ids.get(s.id, "?")
        lines.append(f'    # Generated from {src_id}  ({s.request_type})')
        lines.append('    EvalSample(')
        lines.append(f'        id="{s.id}",')
        lines.append(f'        intent={_py_str(s.intent)},')
        lines.append(f'        scenario={_py_str(s.scenario)},')
        lines.append(f'        expected_class="{s.expected_class}",')
        lines.append(f'        expected_slots={{}},  # REVIEW')
        conflicts_repr = repr(s.expected_conflicts) if s.expected_conflicts else '[]  # REVIEW'
        lines.append(f'        expected_conflicts={conflicts_repr},')
        lines.append(f'        request_type="{s.request_type}",')
        lines.append(f'        scenario_label="{s.scenario_label}",')
        lines.append(f'        gold_deployment_mode="{s.gold_deployment_mode}",')
        lines.append('    ),')
    lines.append(']')
    return "\n".join(lines) + "\n"


def to_json(samples: list[EvalSample], base_ids: dict[str, str]) -> list[dict]:
    return [
        {
            "id":                   s.id,
            "source_id":            base_ids.get(s.id, "?"),
            "request_type":         s.request_type,
            "expected_class":       s.expected_class,
            "scenario_label":       s.scenario_label,
            "gold_deployment_mode": s.gold_deployment_mode,
            "intent":               s.intent,
            "scenario":             s.scenario,
            "expected_slots":       s.expected_slots,
            "expected_conflicts":   s.expected_conflicts,
        }
        for s in samples
    ]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate new EvalSamples from seed intents using the LLM"
    )
    parser.add_argument(
        "--model",
        default="models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--class", dest="filter_class", default=None,
        metavar="PREFIX",
        help="Only process seeds from this class prefix: TC, MC, RC, DC",
    )
    parser.add_argument(
        "--type", dest="variant_type", default=None,
        choices=["paraphrased", "ambiguous", "contradictory"],
        help="Generate only this variant type (default: all three)",
    )
    parser.add_argument(
        "--ids", nargs="+", default=None,
        metavar="ID",
        help="Process specific seed IDs only (e.g. TC-01 MC-03)",
    )
    parser.add_argument(
        "--n", type=int, default=2, dest="n_per_seed",
        metavar="N",
        help="Number of variants to generate per seed per type (default: 2)",
    )
    parser.add_argument(
        "--output", default=None, metavar="FILE",
        help="Override output JSON path (default: eval/generated_samples.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be generated without calling the LLM",
    )
    args = parser.parse_args()

    # Build seed list
    seeds = SEED_SAMPLES
    if args.filter_class:
        seeds = [s for s in seeds if s.id.startswith(args.filter_class)]
    if args.ids:
        seeds = [s for s in seeds if s.id in args.ids]

    if not seeds:
        print("No seed samples matched the filter — nothing to generate.")
        sys.exit(1)

    variant_types = (
        [args.variant_type] if args.variant_type
        else ["paraphrased", "ambiguous", "contradictory"]
    )

    total = len(seeds) * len(variant_types) * args.n_per_seed
    print(
        f"\nDataset Generator — {len(seeds)} seed(s) × {len(variant_types)} type(s)"
        f" × {args.n_per_seed} variant(s) = {total} calls"
    )

    if args.dry_run:
        for s in seeds:
            for vt in variant_types:
                for i in range(args.n_per_seed):
                    print(f"  would generate: {s.id} → {vt} #{i+1}")
        sys.exit(0)

    # Validate model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Pass the correct path with:  --model /path/to/model.gguf")
        sys.exit(1)

    print("Loading LLM model ...")
    llm = LLMClient(model_path=str(model_path))

    # Collect existing IDs to avoid clashes
    existing_ids: set[str] = {s.id for s in ALL_SAMPLES}
    generated: list[EvalSample] = []
    base_ids: dict[str, str] = {}   # new_id → source_id

    # Running counters per (class_prefix, variant_type) for sequential IDs
    counters: dict[str, int] = {}

    for seed in seeds:
        for vt in variant_types:
            for attempt in range(args.n_per_seed):
                counter_key = f"{seed.id.split('-')[0]}-{vt}"
                counters[counter_key] = counters.get(counter_key, 0) + 1
                idx = counters[counter_key]

                new_id = _next_id(seed.id, vt, idx, existing_ids)
                print(
                    f"\n  [{len(generated)+1}/{total}] {seed.id} → {vt} #{attempt+1}"
                    f"  (new id: {new_id})"
                )
                print(f"  Seed: {seed.intent[:80]}...")

                variant = generate_variant(llm, seed, vt)
                if variant is None:
                    print("  SKIPPED — LLM parse failure")
                    continue

                new_sample = build_eval_sample(seed, variant, vt, new_id)
                existing_ids.add(new_id)
                generated.append(new_sample)
                base_ids[new_id] = seed.id

                # Print brief preview
                print(f"  → {new_sample.intent[:100]}...")
                if vt == "contradictory" and new_sample.expected_conflicts:
                    print(f"     conflict: {new_sample.expected_conflicts[0][:80]}")

    if not generated:
        print("\nNo samples generated — all failed.")
        sys.exit(1)

    print(f"\n{len(generated)} sample(s) generated.")

    # Write Python source
    py_src = to_python_source(generated, base_ids)
    OUT_PY.write_text(py_src)
    print(f"\nPython source → {OUT_PY}")
    print(f"  (append GENERATED_SAMPLES to ALL_SAMPLES in dataset.py after review)")

    # Write JSON
    out_json_path = Path(args.output) if args.output else OUT_JSON
    out_json_path.write_text(json.dumps(to_json(generated, base_ids), indent=2))
    print(f"JSON output   → {out_json_path}")
    print(
        "\nIMPORTANT: Review the generated samples before adding them to dataset.py.\n"
        "  Fields marked  # REVIEW  (expected_slots, expected_conflicts) need\n"
        "  manual verification — the LLM does not guarantee correct labels."
    )


if __name__ == "__main__":
    main()
