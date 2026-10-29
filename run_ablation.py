#!/usr/bin/env python3
"""
run_ablation.py — Ablation Study Runner

Runs the full eval over four system variants and reports how each component
contributes to the final metrics.

Variants
--------
  baseline     Full pipeline (S2 extraction + KG conflict detection)
  no_s2        Skip slot extraction — all slots must be elicited via dialogue
  no_kg        Skip KG conflict detection — no conflict flags raised
  no_s2_no_kg  Both disabled — pure dialogue without extraction or conflict checking

Metrics reported per variant: M1–M8

Output
------
  eval/ablation_results.json  — full per-variant, per-sample results
  (table printed to stdout)

Usage
-----
    python run_ablation.py
    python run_ablation.py --class TC
    python run_ablation.py --ids TC-01 CF-01 CF-02
    python run_ablation.py --variants baseline no_s2
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval.dataset import ALL_SAMPLES, EvalSample
from eval.metrics import aggregate, compute_sample_metrics
from ontology.kg import KGOntology
from ontology.ontology_v3 import ALL_CLASSES
from ontology.types import RequiresParameter, RequiresConditional
from pipeline.dialogue_runner import DialogueRunner
from pipeline.llm_client import LLMClient
from pipeline.models import DialogueState, SingleClass, SlotValue

EVAL_DIR    = Path(__file__).parent / "eval"
OUT_PATH    = EVAL_DIR / "ablation_results.json"

ALL_VARIANTS = ["baseline", "no_s2", "no_kg", "no_s2_no_kg", "llm_conflicts"]

VARIANT_LABELS = {
    "baseline":     "Baseline (full system)",
    "no_s2":        "No S2 (no extraction)",
    "no_kg":        "No KG conflicts",
    "no_s2_no_kg":  "No S2 + No KG",
    "llm_conflicts": "LLM conflict det.",
}


# ── Ablation runner ────────────────────────────────────────────────────────────

class AblationRunner(DialogueRunner):
    """
    DialogueRunner subclass with component disabling for ablation experiments.

    no_extraction : bool
        If True, Stage 2 returns an empty state — all slots start as missing
        and must be elicited via dialogue.  Models what happens without the
        slot-extraction component.

    no_conflicts : bool
        If True, the KG conflict-detection call inside Stage 3 is skipped.
        ConflictFlags are never raised regardless of slot values.
        Models what happens without the knowledge-graph conflict-checking layer.
    """

    def __init__(self, *, no_extraction: bool = False, no_conflicts: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.no_extraction = no_extraction
        self.no_conflicts  = no_conflicts

    def _stage2_extract(self, intent: str, class_id: str) -> DialogueState:
        if self.no_extraction:
            # Skip LLM extraction — return state with only implied fields,
            # all mandatory slots still missing
            satisfied: dict[str, SlotValue] = {}
            for field, value in self.kg.get_implied_fields(class_id).items():
                satisfied[field] = SlotValue(
                    field=field, value=value, source="implied", confidence="high"
                )
            missing = list(self.kg.init_missing(class_id))
            if "vertical_domain" not in satisfied:
                missing.insert(0, "vertical_domain")
            return DialogueState(
                intent=intent,
                classification=SingleClass(
                    class_id=class_id,
                    group=self.kg.get_group(class_id),
                    confidence_proxy=0.9,
                ),
                current_class=class_id,
                satisfied=satisfied,
                missing=missing,
                conditionally_activated=[],
            )
        return super()._stage2_extract(intent, class_id)

    def _stage3_elicit(self, intent: str, state: DialogueState) -> DialogueState:
        if not self.no_conflicts:
            return super()._stage3_elicit(intent, state)

        # Replicate Stage 3 loop but skip check_conflicts
        class_id = state.current_class
        history: list[dict[str, str]] = []

        while not state.is_at_turn_limit():
            # ── conflict detection SKIPPED (ablated) ──

            # Activated conditionals still run (they depend on slot values, not conflicts)
            activated = self.kg.get_activated_conditionals(class_id, state.satisfied)
            for cond in activated:
                if cond.field not in state.satisfied and cond.field not in state.missing:
                    state.missing.append(cond.field)
                    state.conditionally_activated.append(cond.field)

            if self.kg.check_termination(class_id, state):
                break

            if not state.missing:
                break

            field = state.missing[0]
            slot_def = self.kg.get_slot_definition(class_id, field)
            if slot_def is None:
                state.missing.remove(field)
                continue

            answer, history = self._ask(slot_def.question, history, state)
            parsed = self.llm.parse_answer(slot_def, answer)
            value = parsed.get("value")
            if value is not None:
                state.satisfy_slot(SlotValue(
                    field=field, value=value,
                    source="user_provided",
                    confidence=parsed.get("confidence", "medium"),
                ))
            else:
                state.missing.remove(field)

        return state


# ── LLM-based conflict detector ───────────────────────────────────────────────

import json as _json
import re as _re

def _llm_check_conflicts(llm: LLMClient, class_id: str, satisfied: dict) -> list:
    """
    Ask the LLM to detect contradictions in the current slot values,
    replacing the KG rule-based check.  Returns ConflictFlag instances.
    """
    from pipeline.models import ConflictFlag

    if not satisfied:
        return []

    slots_str = "\n".join(
        f"  {k}: {v.value}" for k, v in satisfied.items()
    )
    system = (
        "You are a 5G/6G network configuration validator. "
        "Given a service class and a set of collected parameters, identify any "
        "contradictory or mutually incompatible combinations. "
        "Examples of conflicts: Critical priority with LowCost profile, "
        "Dedicated slice with LowCost profile, GEO satellite with Critical priority, "
        "Maritime scenario with TN-only deployment mode. "
        "Respond with a JSON object only — no prose, no markdown fences."
    )
    user = (
        f"Service class: {class_id}\n"
        f"Collected parameters:\n{slots_str}\n\n"
        "Are there any contradictory or incompatible parameter combinations?\n"
        'Return JSON: {"has_conflict": true/false, "conflicts": [{"field_a": "...", "field_b": "...", "description": "..."}]}'
    )

    model = llm._get_model()
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=256,
        temperature=0.0,
    )
    text = response["choices"][0]["message"]["content"].strip()

    try:
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if not m:
            return []
        data = _json.loads(m.group(0))
        if not data.get("has_conflict"):
            return []
        flags = []
        for c in data.get("conflicts", []):
            flags.append(ConflictFlag(
                conflict_type="llm_detected",
                description=c.get("description", "LLM-detected conflict"),
                subject_field=c.get("field_a", ""),
                object_field=c.get("field_b", ""),
            ))
        return flags
    except Exception:
        return []


class LLMConflictRunner(DialogueRunner):
    """
    DialogueRunner that replaces KG conflict detection with an LLM call.
    Everything else (extraction, elicitation loop, assembly) is identical
    to the baseline.  Lets us compare: KG rules vs. LLM as conflict detector.
    """

    def _stage3_elicit(self, intent: str, state: DialogueState) -> DialogueState:
        class_id = state.current_class
        history: list[dict[str, str]] = []

        while not state.is_at_turn_limit():
            # LLM-based conflict detection (replaces kg.check_conflicts)
            new_conflicts = _llm_check_conflicts(self.llm, class_id, state.satisfied)
            for flag in new_conflicts:
                state.add_conflict(flag)

            activated = self.kg.get_activated_conditionals(class_id, state.satisfied)
            for cond in activated:
                if cond.field not in state.satisfied and cond.field not in state.missing:
                    state.missing.append(cond.field)
                    state.conditionally_activated.append(cond.field)

            if self.kg.check_termination(class_id, state):
                break

            unresolved = [c for c in state.conflicts if not c.resolved]
            if unresolved:
                conflict = unresolved[0]
                q = (
                    f"There is a conflict: {conflict.description} "
                    f"Please choose: keep '{conflict.subject_field}' as is "
                    f"or change '{conflict.object_field}'?"
                )
                answer, history = self._ask(q, history, state)
                from pipeline.dialogue_runner import _synthetic_slot
                parsed = self.llm.parse_answer(
                    _synthetic_slot(conflict.object_field), answer
                )
                if parsed.get("value"):
                    state.satisfy_slot(SlotValue(
                        field=conflict.object_field,
                        value=parsed["value"],
                        source="user_provided",
                        confidence=parsed.get("confidence", "medium"),
                    ))
                state.resolve_conflict(conflict.description, answer)
                continue

            if not state.missing:
                break

            field = state.missing[0]
            slot_def = self.kg.get_slot_definition(class_id, field)
            if slot_def is None:
                state.missing.remove(field)
                continue

            answer, history = self._ask(slot_def.question, history, state)
            parsed = self.llm.parse_answer(slot_def, answer)
            value = parsed.get("value")
            if value is not None:
                state.satisfy_slot(SlotValue(
                    field=field, value=value,
                    source="user_provided",
                    confidence=parsed.get("confidence", "medium"),
                ))
            else:
                state.missing.remove(field)

        return state


# ── Per-variant eval ───────────────────────────────────────────────────────────

def run_variant(
    samples: list[EvalSample],
    variant: str,
    llm: LLMClient,
    kg: KGOntology,
    verbose: bool,
) -> list:
    """Run all samples for one ablation variant. Returns list of SampleResult."""
    no_extraction = variant in ("no_s2", "no_s2_no_kg")
    no_conflicts  = variant in ("no_kg", "no_s2_no_kg")

    results = []
    for i, sample in enumerate(samples, 1):
        print(f"    [{i}/{len(samples)}] {sample.id}", end="", flush=True)
        try:
            if variant == "llm_conflicts":
                runner = LLMConflictRunner(
                    scenario=sample.scenario,
                    llm=llm,
                    kg=kg,
                    verbose=verbose,
                )
            else:
                runner = AblationRunner(
                    scenario=sample.scenario,
                    llm=llm,
                    kg=kg,
                    verbose=verbose,
                    no_extraction=no_extraction,
                    no_conflicts=no_conflicts,
                )
            si, state = runner.run(sample.intent)
            result = compute_sample_metrics(
                sample_id=sample.id,
                expected_class=sample.expected_class,
                expected_conflict_keywords=sample.expected_conflicts,
                request_type=sample.request_type,
                si=si,
                state=state,
                kg=kg,
            )
            print(
                f"  cls:{'Y' if result.classification_correct else 'N'}"
                f"  cov:{result.extraction_coverage:.0%}"
                f"  turns:{result.turns_to_termination}"
                f"  M8:{'Y' if result.detected_conflicts > 0 else '-'}"
            )
            results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if verbose:
                traceback.print_exc()

    return results


# ── Output helpers ─────────────────────────────────────────────────────────────

def _agg_dict(agg) -> dict:
    return {
        "n_samples":                          agg.n_samples,
        "M1_classification_accuracy":         agg.classification_accuracy,
        "M2_avg_extraction_coverage":         agg.avg_extraction_coverage,
        "M3_avg_turns":                       agg.avg_turns,
        "M4_avg_profile_completeness":        agg.avg_profile_completeness,
        "M5_conflict_detection_rate":         agg.overall_conflict_detection_rate,
        "M6_proper_termination_rate":         agg.proper_termination_rate,
        "M7_policy_validity_rate":            agg.policy_validity_rate,
        "M8_contradiction_detection_accuracy": agg.contradiction_detection_accuracy,
    }


def print_ablation_table(variant_aggs: dict[str, object]) -> None:
    metrics = [
        ("M1_classification_accuracy",          "M1  Classification Acc.",       True),
        ("M2_avg_extraction_coverage",           "M2  Extraction Coverage",       True),
        ("M3_avg_turns",                         "M3  Avg Turns",                 False),
        ("M4_avg_profile_completeness",          "M4  Profile Completeness",      True),
        ("M5_conflict_detection_rate",           "M5  Conflict Detection Rate",   True),
        ("M6_proper_termination_rate",           "M6  Proper Termination Rate",   True),
        ("M7_policy_validity_rate",              "M7  Policy Validity Rate",      True),
        ("M8_contradiction_detection_accuracy",  "M8  Contradiction Det. Acc.",   True),
    ]

    col_w = 14
    header_variants = list(variant_aggs.keys())
    col_headers = [VARIANT_LABELS[v][:col_w] for v in header_variants]

    print("\n" + "=" * (42 + col_w * len(header_variants)))
    print("ABLATION STUDY RESULTS")
    print("=" * (42 + col_w * len(header_variants)))
    print(f"{'Metric':<42}" + "".join(f"{h:>{col_w}}" for h in col_headers))
    print("-" * (42 + col_w * len(header_variants)))

    agg_dicts = {v: _agg_dict(agg) for v, agg in variant_aggs.items()}

    # Load baseline from saved file if not in the current run
    saved_baseline = None
    if "baseline" not in agg_dicts and OUT_PATH.exists():
        try:
            saved = json.loads(OUT_PATH.read_text())
            if "baseline" in saved:
                saved_baseline = saved["baseline"]["aggregate"]
        except Exception:
            pass

    for key, label, is_rate in metrics:
        row = f"  {label:<40}"
        if "baseline" in agg_dicts:
            baseline_val = agg_dicts["baseline"].get(key)
        elif saved_baseline:
            baseline_val = saved_baseline.get(key)
        else:
            baseline_val = None
        for v in header_variants:
            val = agg_dicts[v].get(key)
            if isinstance(val, float):
                fmt = ".1%" if is_rate else ".2f"
                cell = f"{val:{fmt}}"
                if v != "baseline" and isinstance(baseline_val, float):
                    delta = val - baseline_val
                    dfmt = "+.1%" if is_rate else "+.2f"
                    cell += f"({delta:{dfmt}})"
            elif isinstance(val, int):
                cell = str(val)
            else:
                cell = "—"
            row += f"{cell:>{col_w}}"
        print(row)

    print("=" * (42 + col_w * len(header_variants)))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Intent Agent — ablation study")
    parser.add_argument("--model", default="models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")
    parser.add_argument("--class", dest="filter_class", default=None, metavar="PREFIX")
    parser.add_argument("--type", dest="filter_type", default=None,
                        choices=["seed", "paraphrased", "ambiguous", "contradictory"])
    parser.add_argument("--ids", nargs="+", default=None, metavar="ID")
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                        choices=ALL_VARIANTS, metavar="VARIANT",
                        help=f"Variants to run (default: all). Choices: {ALL_VARIANTS}")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Apply sample filters
    samples = ALL_SAMPLES
    if args.filter_class:
        samples = [s for s in samples if s.id.startswith(args.filter_class)]
    if args.filter_type:
        samples = [s for s in samples if s.request_type == args.filter_type]
    if args.ids:
        samples = [s for s in samples if s.id in args.ids]

    if not samples:
        print("No samples matched — nothing to run.")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"\nIntent Agent — Ablation Study")
    print(f"  Samples : {len(samples)}")
    print(f"  Variants: {args.variants}")
    print(f"  Total   : {len(samples) * len(args.variants)} inference runs\n")

    print("Loading LLM model ...")
    llm = LLMClient(model_path=str(model_path))
    kg  = KGOntology(ALL_CLASSES)

    variant_aggs:    dict[str, object] = {}
    variant_results: dict[str, list]   = {}

    for variant in args.variants:
        print(f"\n── Variant: {VARIANT_LABELS[variant]} ──")
        results = run_variant(samples, variant, llm, kg, args.verbose)
        if results:
            agg = aggregate(results)
            variant_aggs[variant]    = agg
            variant_results[variant] = results

    if not variant_aggs:
        print("All variants failed.")
        sys.exit(1)

    # Save JSON first (before printing, so results are never lost on crash)
    output = {
        variant: {
            "aggregate": _agg_dict(agg),
            "samples": [
                {
                    "id":                    r.sample_id,
                    "expected_class":        r.expected_class,
                    "predicted_class":       r.predicted_class,
                    "request_type":          r.request_type,
                    "classification_correct": r.classification_correct,
                    "extraction_coverage":   r.extraction_coverage,
                    "turns":                 r.turns_to_termination,
                    "profile_completeness":  r.profile_completeness,
                    "conflict_detection_rate": r.conflict_detection_rate,
                    "proper_termination":    r.proper_termination,
                    "policy_valid":          r.policy_valid,
                    "detected_conflicts":    r.detected_conflicts,
                }
                for r in variant_results[variant]
            ],
        }
        for variant, agg in variant_aggs.items()
    }
    # Merge with any previously saved variants so the file stays cumulative
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text())
            existing.update(output)
            output = existing
        except Exception:
            pass

    EVAL_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {OUT_PATH}")

    print_ablation_table(variant_aggs)


if __name__ == "__main__":
    main()
