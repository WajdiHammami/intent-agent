#!/usr/bin/env python3
"""
run_pure_llm.py — Pure-LLM Pipeline Evaluation

Replaces the Knowledge Graph reasoning layer (Stage 3) entirely with LLM
calls. The LLM is responsible for:

  - Determining which required attributes are still missing
  - Detecting incompatible value combinations (conflicts)
  - Generating clarification questions
  - Deciding when the profile is complete
  - Assigning deployment mode (Stage 6)

Everything the KG does via programmatic rules is instead encoded in
LLM prompts. This isolates the KG's contribution: any metric gap between
this runner and the baseline KG runner is directly attributable to the
KG reasoning layer.

What is NOT changed vs the baseline:
  - Stage 1 (intent classification) — identical LLM call
  - Stage 2 (slot extraction)       — identical LLM call
  - Evaluation metrics              — identical computation (KG used only
                                      to CHECK the output, not produce it)

Output
------
  eval/pure_llm_results.json   — per-sample results
  Printed comparison table vs KG baseline (loaded from ablation_results.json
  if available, otherwise just the pure-LLM numbers).

Usage
-----
    python run_pure_llm.py
    python run_pure_llm.py                     # full 72-sample set
    python run_pure_llm.py --contradictory-only  # 40-sample contradictory subset
    python run_pure_llm.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from eval.dataset import ALL_SAMPLES, CONTRADICTORY_SUBSET, EvalSample
from eval.metrics import aggregate, compute_sample_metrics, print_results_table
from ontology.kg import KGOntology
from ontology.ontology_v3 import ALL_CLASSES
from pipeline.dialogue_runner import DialogueRunner, _synthetic_slot
from pipeline.llm_client import LLMClient, _chat, _extract_json
from pipeline.models import ConflictFlag, DialogueState, ServiceIntent, SlotValue

EVAL_DIR = Path(__file__).parent / "eval"
OUT_PATH = EVAL_DIR / "pure_llm_results.json"


# ── LLM domain knowledge (replaces KG schema) ─────────────────────────────────

_CLASS_INFO: dict[str, dict] = {
    "TimeCritical": {
        "description": (
            "Ultra-low latency industrial/vehicular control (≤1 ms, 99.9999% reliability). "
            "Robotics, grid protection, V2X platooning. Always Dedicated slice, Critical priority."
        ),
        "required_fields": [
            "vertical_domain", "deployment_scenario", "geographic_scope",
            "cost_profile", "application_type", "cycle_time_ms",
        ],
        "implied": {"priority_level": "Critical", "isolation_level": "Dedicated"},
    },
    "MissionCritical": {
        "description": (
            "Resilience-first connectivity (≤10 ms, 99.999% availability). "
            "Public safety, emergency response, critical infrastructure. "
            "Mandatory TN/NTN fallback support."
        ),
        "required_fields": [
            "vertical_domain", "deployment_scenario", "geographic_scope",
            "isolation_level", "cost_profile", "deployment_mode",
        ],
        "conditional_note": (
            "If deployment_mode is Hybrid or NTN_Only, also ask: "
            "ntn_tier (LEO/GEO/UAV_HAPS) and max_failover_sec."
        ),
        "implied": {"priority_level": "High"},
    },
    "RemoteConnectivity": {
        "description": (
            "NTN-primary (satellite/UAV) coverage for areas without terrestrial networks "
            "(≤500 ms, 99.9% availability). Maritime, rural, aerial, polar."
        ),
        "required_fields": [
            "vertical_domain", "deployment_scenario", "geographic_scope",
            "cost_profile", "ntn_tier", "min_throughput_mbps",
        ],
        "conditional_note": (
            "If the service involves voice/video/real-time interaction, "
            "also ask: interactive_traffic (True/False)."
        ),
        "implied": {"priority_level": "Normal"},
    },
    "MassiveConnectivity": {
        "description": (
            "Massive numbers of low-power IoT devices (≥10⁶ dev/km², 1 Kbps/device). "
            "Smart metering, industrial sensors, precision agriculture."
        ),
        "required_fields": [
            "vertical_domain", "deployment_scenario", "geographic_scope",
            "cost_profile", "device_density_per_km2",
        ],
        "conditional_note": (
            "Also useful: reporting_interval_sec, payload_bytes, battery (True/False), "
            "target_lifetime_years."
        ),
        "implied": {"priority_level": "Low"},
    },
}

_CONFLICT_RULES = """\
Known incompatible value combinations (flag ALL that apply):
  1. priority_level=Critical  + cost_profile=LowCost
       → Critical SLA cannot be guaranteed at LowCost; use Balanced or Premium.
  2. isolation_level=Dedicated + cost_profile=LowCost
       → Dedicated slices are not offered at LowCost; use Balanced or Premium.
  3. deployment_scenario=Maritime + deployment_mode=TN_Only
       → No terrestrial network in Maritime; use NTN_Only or Hybrid.
  4. ntn_tier=GEO + priority_level=Critical
       → GEO RTT ~540 ms cannot meet Critical latency requirements; use LEO.
  5. deployment_scenario=Disaster + deployment_mode=TN_Only
       → TN infrastructure may be degraded/destroyed in Disaster; use Hybrid or NTN_Only.
  6. ntn_tier=GEO + (interactive_traffic=True OR any latency requirement ≤ 200 ms)
       → GEO RTT ~540 ms is incompatible with interactive or latency-sensitive applications.
  7. isolation_level=Dedicated + cost_profile=Premium + class=MassiveConnectivity
       → Dedicated Premium slices are economically unviable for massive IoT scale."""


# ── PureLLMRunner ─────────────────────────────────────────────────────────────

class PureLLMRunner(DialogueRunner):
    """
    Pipeline variant that replaces the KG reasoning layer with LLM calls.

    Stage 3 (elicitation loop) is fully overridden. The LLM uses its own
    domain knowledge to determine:
      - which required fields are missing
      - which value combinations are conflicting
      - when the profile is complete
      - what question to ask next

    Stages 1 and 2 are inherited unchanged (classification + slot extraction).
    Stage 5/6 is inherited but deployment mode is assigned by LLM instead of KG.
    """

    def _stage3_elicit(self, intent: str, state: DialogueState) -> DialogueState:
        history: list[dict[str, str]] = []

        while not state.is_at_turn_limit():
            # ── Single LLM call: assess completeness + conflicts ──────────────
            assessment = self._llm_assess(state)

            # Register any new conflicts reported by the LLM
            for c in assessment.get("conflicts", []):
                desc = c.get("description", "").strip()
                if not desc:
                    continue
                flag = ConflictFlag(
                    conflict_type="kpi_conflict",
                    description=desc,
                    subject_field=c.get("field_a", ""),
                    object_field=c.get("field_b", ""),
                )
                state.add_conflict(flag)  # dedup is handled in add_conflict()

            # ── Resolve unresolved conflicts first ────────────────────────────
            unresolved = [c for c in state.conflicts if not c.resolved]
            if unresolved:
                conflict = unresolved[0]
                q = (
                    f"There is a conflict: {conflict.description} "
                    f"Please choose a way to resolve it — which value would you like to change?"
                )
                answer, history = self._ask(q, history, state)
                resolved = self._llm_parse_resolution(conflict, answer)
                if resolved.get("field") and resolved.get("value") is not None:
                    state.satisfy_slot(SlotValue(
                        field=resolved["field"],
                        value=resolved["value"],
                        source="user_provided",
                        confidence="medium",
                    ))
                state.resolve_conflict(conflict.description, answer)
                continue

            # ── Termination check ─────────────────────────────────────────────
            if assessment.get("is_complete", False):
                break

            missing = assessment.get("missing_fields", [])
            if not missing:
                break

            # ── Ask about the next missing field ──────────────────────────────
            field = missing[0]
            questions = assessment.get("questions", {})
            question = questions.get(
                field,
                f"What is the {field.replace('_', ' ')} for this service deployment?",
            )
            answer, history = self._ask(question, history, state)

            value = self._llm_parse_field(field, state.current_class, answer)
            if value is not None:
                state.satisfy_slot(SlotValue(
                    field=field,
                    value=value,
                    source="user_provided",
                    confidence="medium",
                ))
            else:
                # Could not parse — skip this field to avoid infinite loop
                pass  # LLM will re-assess next iteration

        return state

    # ── LLM reasoning calls ───────────────────────────────────────────────────

    def _llm_assess(self, state: DialogueState) -> dict:
        """
        Single LLM call that returns:
          - is_complete: bool
          - missing_fields: list of field names still needed
          - questions: dict mapping field → natural-language question
          - conflicts: list of {field_a, value_a, field_b, value_b, description}
        """
        class_id = state.current_class
        info = _CLASS_INFO.get(class_id, {})
        implied = info.get("implied", {})

        filled_lines = [
            f"  {k}: {v.value}" for k, v in state.satisfied.items()
        ]
        filled_str = "\n".join(filled_lines) if filled_lines else "  (none yet)"
        required_str = ", ".join(info.get("required_fields", []))
        conditional_note = info.get("conditional_note", "")

        system = (
            "You are an expert 6G service intent validator with knowledge of "
            "TN/NTN network requirements. Analyze the service profile and return "
            "a JSON assessment. Be concise. Return ONLY the JSON object."
        )
        user = f"""Service class: {class_id}
Class description: {info.get("description", "")}
Original intent: "{state.intent}"

Currently satisfied attributes:
{filled_str}

Required fields for {class_id}: {required_str}
{f"Additional conditional fields: {conditional_note}" if conditional_note else ""}
Implied fields (already set, do NOT ask): {implied}

{_CONFLICT_RULES}

Assess the current profile. Return JSON:
{{
  "is_complete": <true if all required fields are present AND no unresolved conflicts>,
  "missing_fields": ["<field_name>", ...],
  "questions": {{
    "<field_name>": "<concise natural-language question to ask the user>"
  }},
  "conflicts": [
    {{
      "field_a": "<field>",
      "value_a": "<value>",
      "field_b": "<field>",
      "value_b": "<value>",
      "description": "<one-sentence conflict explanation>"
    }}
  ]
}}"""

        try:
            text = _chat(self.llm._get_model(), system, user, max_tokens=600)
            return _extract_json(text)
        except Exception:
            return {
                "is_complete": False,
                "missing_fields": [],
                "questions": {},
                "conflicts": [],
            }

    def _llm_parse_field(self, field: str, class_id: str, answer: str) -> Any:
        """Parse a free-text user answer into a typed value for the given field."""
        system = (
            "You are a value extractor for a 6G network configuration system. "
            "Extract the typed value from the user's answer. "
            "Return ONLY a JSON object."
        )
        user = (
            f'Service class: {class_id}\n'
            f'Field: "{field}"\n'
            f'User answer: "{answer}"\n\n'
            'Return JSON: {"value": <extracted typed value or null if unclear>}'
        )
        try:
            text = _chat(self.llm._get_model(), system, user, max_tokens=64)
            result = _extract_json(text)
            return result.get("value")
        except Exception:
            return None

    def _llm_parse_resolution(self, conflict: ConflictFlag, answer: str) -> dict:
        """
        Parse which field+value the user chose to change in order to resolve
        a conflict. Returns {"field": ..., "value": ...}.
        """
        system = (
            "You are a conflict resolution parser for a 6G configuration system. "
            "Return ONLY a JSON object."
        )
        user = (
            f'Conflict: "{conflict.description}"\n'
            f'Fields involved: "{conflict.subject_field}", "{conflict.object_field}"\n'
            f'User response: "{answer}"\n\n'
            'Which field did the user agree to change, and to what new value? '
            'Return JSON: {"field": "<field_name>", "value": "<new_value>"}'
        )
        try:
            text = _chat(self.llm._get_model(), system, user, max_tokens=64)
            return _extract_json(text)
        except Exception:
            return {}

    def _stage5_assemble(self, state: DialogueState) -> ServiceIntent:
        """Override: assign deployment mode via LLM instead of KG lookup."""
        if "deployment_mode" not in state.satisfied:
            mode = self._llm_assign_deployment_mode(state)
            state.satisfy_slot(SlotValue(
                field="deployment_mode",
                value=mode,
                source="user_provided",
                confidence="medium",
            ))
            self._log(f"[S6-LLM] Deployment mode assigned by LLM: {mode}")
        return super()._stage5_assemble(state)

    def _llm_assign_deployment_mode(self, state: DialogueState) -> str:
        """Ask the LLM to recommend a deployment mode based on class + context."""
        class_id = state.current_class
        scenario_sv = state.satisfied.get("deployment_scenario")
        scenario = str(scenario_sv.value) if scenario_sv else "Unknown"
        filled = {k: str(v.value) for k, v in state.satisfied.items()}

        system = (
            "You are a 6G deployment mode recommender. "
            "Choose the most appropriate deployment mode. Return ONLY JSON."
        )
        user = (
            f"Service class: {class_id}\n"
            f"Deployment scenario: {scenario}\n"
            f"Satisfied attributes: {json.dumps(filled)}\n\n"
            "Choose deployment mode — TN_Only, NTN_Only, or Hybrid — using these guidelines:\n"
            "  - Urban + TimeCritical/MissionCritical: TN_Only or Hybrid\n"
            "  - Maritime or Remote + RemoteConnectivity/MassiveConnectivity: NTN_Only or Hybrid\n"
            "  - Disaster scenario: Hybrid (TN may be degraded)\n"
            "  - RuralRemote + any class: Hybrid or NTN_Only\n"
            "  - MissionCritical always needs fallback: Hybrid unless explicitly NTN_Only\n"
            'Return JSON: {"deployment_mode": "TN_Only|NTN_Only|Hybrid"}'
        )
        try:
            text = _chat(self.llm._get_model(), system, user, max_tokens=64)
            result = _extract_json(text)
            mode = result.get("deployment_mode", "Hybrid")
            if mode not in ("TN_Only", "NTN_Only", "Hybrid"):
                mode = "Hybrid"
            return mode
        except Exception:
            return "Hybrid"


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_sample(
    sample: EvalSample,
    llm: LLMClient,
    kg: KGOntology,
    verbose: bool,
):
    runner = PureLLMRunner(
        scenario=sample.scenario,
        llm=llm,
        kg=kg,
        verbose=verbose,
    )
    si, state = runner.run(sample.intent)
    return compute_sample_metrics(
        sample_id=sample.id,
        expected_class=sample.expected_class,
        expected_conflict_keywords=sample.expected_conflicts,
        request_type=sample.request_type,
        si=si,
        state=state,
        kg=kg,
    )


def print_comparison_table(pure_agg, baseline_path: Path | None = None) -> None:
    """Print pure-LLM results and optionally compare with KG baseline."""
    baseline = None
    if baseline_path and baseline_path.exists():
        try:
            data = json.loads(baseline_path.read_text())
            baseline = data.get("baseline", {})
        except Exception:
            pass

    metrics = [
        ("M1 Classification Accuracy", "classification_accuracy",         True),
        ("M3 Avg Turns",               "avg_turns",                       False),
        ("M7 Policy Validity Rate",    "policy_validity_rate",             True),
        ("M8 Contradiction Det. Acc.", "contradiction_detection_accuracy", True),
    ]

    col = 32
    print("\n" + "=" * (col + 40))
    print("PURE LLM vs KG BASELINE")
    print("=" * (col + 40))
    hdr = f"{'Metric':<{col}} {'Pure-LLM':>12}"
    if baseline:
        hdr += f" {'KG-Baseline':>12} {'Δ':>8}"
    print(hdr)
    print("-" * (col + 40))

    def _agg_val(agg, key):
        if hasattr(agg, key):
            return getattr(agg, key)
        if isinstance(agg, dict):
            return agg.get(key)
        return None

    for label, key, is_rate in metrics:
        pure_val = _agg_val(pure_agg, key)
        row = f"  {label:<{col-2}}"
        if isinstance(pure_val, float):
            row += f" {pure_val:.1%}" .rjust(12) if is_rate else f" {pure_val:.2f}".rjust(12)
        else:
            row += f" {pure_val}".rjust(12)

        if baseline:
            base_val = _agg_val(baseline, key)
            if isinstance(base_val, float) and isinstance(pure_val, float):
                row += f" {base_val:.1%}".rjust(12) if is_rate else f" {base_val:.2f}".rjust(12)
                delta = pure_val - base_val
                sign = "+" if delta >= 0 else ""
                row += f" {sign}{delta:.1%}".rjust(8) if is_rate else f" {sign}{delta:.2f}".rjust(8)
            else:
                row += f" {'—':>12}"
        print(row)

    print("=" * (col + 40))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intent Agent — Pure-LLM pipeline (no KG reasoning)"
    )
    parser.add_argument("--model", default="models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")
    parser.add_argument(
        "--contradictory-only", action="store_true",
        help="Run only on the 40-sample contradictory subset (used for the "
             "KG-vs-Pure-LLM comparison in the paper)",
    )
    parser.add_argument("--class", dest="filter_class", default=None, metavar="PREFIX")
    parser.add_argument("--ids", nargs="+", default=None, metavar="ID")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Select dataset: full 72-sample set, or the 40-sample contradictory subset
    if args.contradictory_only:
        samples = list(CONTRADICTORY_SUBSET)
    else:
        samples = list(ALL_SAMPLES)
    if args.filter_class:
        samples = [s for s in samples if s.id.startswith(args.filter_class)]
    if args.ids:
        samples = [s for s in samples if s.id in args.ids]

    if not samples:
        print("No samples matched — nothing to run.")
        sys.exit(1)

    print(f"\nIntent Agent — Pure-LLM Pipeline")
    subset = "contradictory-40" if args.contradictory_only else "main-72"
    print(f"  Dataset  : {subset} ({len(samples)} samples)")
    print(f"  Strategy : No KG reasoning — LLM handles completeness + conflict detection")
    print(f"  Model    : {args.model}\n")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print("Loading LLM model...")
    llm = LLMClient(model_path=str(model_path))
    kg = KGOntology(ALL_CLASSES)  # used only for metric computation, not reasoning

    results = []
    for i, sample in enumerate(samples, 1):
        print(f"  [{i:>3}/{len(samples)}] {sample.id:<14}", end="", flush=True)
        try:
            result = run_sample(sample, llm, kg, args.verbose)
            print(
                f"  M1:{'Y' if result.classification_correct else 'N'}"
                f"  M7:{'Y' if result.policy_valid else 'N'}"
                f"  M8:{'Y' if result.detected_conflicts > 0 else 'N'}"
                f"  turns:{result.turns_to_termination}"
            )
            results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if args.verbose:
                traceback.print_exc()

    if not results:
        print("All samples failed.")
        sys.exit(1)

    agg = aggregate(results)
    print_results_table(results, agg)

    # Print comparison with KG baseline (from ablation_results.json if present)
    baseline_path = EVAL_DIR / "ablation_results.json"
    print_comparison_table(agg, baseline_path)

    # Save results
    output = {
        "variant": "pure_llm",
        "dataset": "contradictory-40" if args.contradictory_only else "main-72",
        "n_samples": len(results),
        "aggregate": {
            "M1_classification_accuracy":          agg.classification_accuracy,
            "M3_avg_turns":                        agg.avg_turns,
            "M7_policy_validity_rate":             agg.policy_validity_rate,
            "M8_contradiction_detection_accuracy": agg.contradiction_detection_accuracy,
        },
        "samples": [
            {
                "id":                      r.sample_id,
                "expected_class":          r.expected_class,
                "predicted_class":         r.predicted_class,
                "request_type":            r.request_type,
                "classification_correct":  r.classification_correct,
                "turns":                   r.turns_to_termination,
                "policy_valid":            r.policy_valid,
                "detected_conflicts":      r.detected_conflicts,
                "conflict_detection_rate": r.conflict_detection_rate,
            }
            for r in results
        ],
    }
    EVAL_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
