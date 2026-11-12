"""
Dataset generator — the LLM proposes paraphrased / ambiguous / contradictory
variants of seed intents. Output is JSON in the dataset's own format; review
the labels before adding rows to ``eval/dataset/data/<type>.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import Settings
from eval.dataset import ALL_SAMPLES, select
from eval.experiments.common import connect
from eval.experiments.generate import prompts
from eval.experiments.generate.builder import build_sample, next_id
from eval.harness import write_json

NAME = "generate"
HELP = "LLM-assisted dataset expansion (labels need human review)"
VARIANT_TYPES = ["paraphrased", "ambiguous", "contradictory"]


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--class", dest="cls", metavar="CLASS", help="seed class: TC, MC, RC, MaC")
    parser.add_argument("--type", dest="variant_type", choices=VARIANT_TYPES,
                        help="only this variant type (default: all three)")
    parser.add_argument("--ids", nargs="+", metavar="ID", help="only these seed ids")
    parser.add_argument("--n", type=int, default=2, dest="n_per_seed", help="variants per seed and type")
    parser.add_argument("--output", type=Path, help="output JSON (default: results/generated_samples.json)")
    parser.add_argument("--dry-run", action="store_true", help="list what would be generated")


def run(args: argparse.Namespace, settings: Settings, llm=None) -> int:
    seeds = select(ALL_SAMPLES, cls=args.cls, rtype="seed", ids=args.ids)
    types = [args.variant_type] if args.variant_type else VARIANT_TYPES
    if not seeds:
        print("No seed samples matched the filters.")
        return 1
    print(f"\nDataset generator — {len(seeds)} seeds × {len(types)} types × {args.n_per_seed}")
    if args.dry_run:
        for s in seeds:
            for vt in types:
                print(f"  would generate {args.n_per_seed} × {vt} from {s.id}")
        return 0

    llm = llm or connect(settings)
    taken = {s.id for s in ALL_SAMPLES}
    counters: dict[str, int] = {}
    generated = []
    for seed in seeds:
        for vt in types:
            for _ in range(args.n_per_seed):
                key = f"{seed.id.split('-')[0]}-{vt}"
                counters[key] = counters.get(key, 0) + 1
                new_id = next_id(seed.id, vt, counters[key], taken)
                try:
                    variant = llm.chat_json(prompts.SYSTEM[vt], prompts.user_prompt(seed.intent, vt),
                                            max_tokens=512, temperature=0.7)
                    sample = build_sample(seed, variant, vt, new_id)
                except (ValueError, KeyError) as exc:
                    print(f"  {seed.id} → {vt}: skipped ({exc})")
                    continue
                taken.add(new_id)
                generated.append({**sample.to_dict(), "source_id": seed.id})
                print(f"  {seed.id} → {new_id}: {sample.intent[:90]}")

    if not generated:
        print("No samples generated.")
        return 1
    path = write_json(args.output or settings.results_dir / "generated_samples.json", generated)
    print(f"\n{len(generated)} sample(s) → {path}\n"
          "Review expected_slots / expected_conflicts before adding them to the dataset.")
    return 0
