"""Run all scenario/graph experiments and persist comparison outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from crabpath.simulator import load_scenarios


def _discover_scenarios(scenarios_dir: Path) -> list[Path]:
    return sorted(path for path in scenarios_dir.glob("*.jsonl"))


def _discover_builders(experiments_dir: Path) -> dict[str, Path]:
    build_map: dict[str, Path] = {}
    for path in sorted(experiments_dir.glob("build_*.py")):
        build_map[path.stem.removeprefix("build_")] = path
    return build_map


def _load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load python module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_build_module(path: Path):
    module = _load_module_from_path(path)
    if not hasattr(module, "build_graph"):
        raise RuntimeError(f"build script missing build_graph(): {path}")
    return module.build_graph


def _build_graph_from_script(path: Path):
    builder = _load_build_module(path)
    return builder()


def _run_single_experiment(
    scenario_path: Path,
    build_path: Path,
    top_k: int,
    max_hops: int,
    comparison_module,
) -> dict:
    scenarios = load_scenarios(scenario_path)
    if not scenarios:
        return {
            "experiment": scenario_path.stem,
            "arms": {arm: {"episodes": [], "avg_tokens": 0.0, "accuracy": 0.0} for arm in (
                "static",
                "rag",
                "crabpath_corrected",
                "crabpath_myopic",
            )},
        }

    graph = _build_graph_from_script(build_path)
    return comparison_module.run_comparison(
        graph=graph,
        scenario_file=scenario_path,
        top_k=top_k,
        max_hops=max_hops,
    )


def _print_summary(all_results: list[dict]) -> None:
    if not all_results:
        print("No experiments were run.")
        return

    print("\nExperiment summary:")
    for result in all_results:
        print(f"\n{result['experiment']}")
        print("  Arm                 Accuracy   AvgTokens")
        print("  ------------------------------------")
        for arm_name, payload in result["arms"].items():
            print(
                f"  {arm_name:19} {payload['accuracy']:.3f}     "
                f"{payload['avg_tokens']:.1f}"
            )


def _load_comparison_module(experiments_dir: Path):
    return _load_module_from_path(experiments_dir / "run_comparison.py")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all configured experiments.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument(
        "--results-dir",
        default="experiments/results",
        help="Directory to write per-scenario result JSON.",
    )
    parser.add_argument("--experiment", default=None, help="Run only this scenario stem.")

    args = parser.parse_args(argv)

    experiments_dir = Path(__file__).resolve().parent
    scenarios_dir = experiments_dir.parent / "scenarios"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    comparison = _load_comparison_module(experiments_dir)
    scenarios = _discover_scenarios(scenarios_dir)
    build_map = _discover_builders(experiments_dir)
    if args.experiment:
        scenarios = [path for path in scenarios if path.stem == args.experiment]

    if not scenarios:
        print("No scenarios found.")
        return 0

    outputs: list[tuple[Path, dict]] = []
    for scenario_path in scenarios:
        build_path = build_map.get(scenario_path.stem)
        if build_path is None:
            print(
                f"Skipping {scenario_path.name}: "
                f"no experiments/build_{scenario_path.stem}.py found."
            )
            continue

        result = _run_single_experiment(
            scenario_path=scenario_path,
            build_path=build_path,
            top_k=args.top_k,
            max_hops=args.max_hops,
            comparison_module=comparison,
        )
        outputs.append((scenario_path, result))

        out_path = results_dir / f"{scenario_path.stem}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    _print_summary([result for _, result in outputs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
