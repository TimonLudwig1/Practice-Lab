"""Run the seeded MST comparison and write its artifacts."""

from __future__ import annotations

from pathlib import Path

from mst_workshop import BenchmarkResult, run_benchmark, write_benchmark_csv


def write_report(results: tuple[BenchmarkResult, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Seeded MST Workshop Results",
        "",
        "Kruskal and Prim were executed on the exact same immutable edge tuple "
        "for every row. Median runtimes use seven repetitions.",
        "",
        "| Vertices | Edges | Seed | MST weight | Kruskal (µs) | Prim (µs) | Same edges |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.vertex_count} | {result.edge_count} | {result.seed} | "
            f"{result.mst_weight:.3f} | {result.kruskal_median_us:.3f} | "
            f"{result.prim_median_us:.3f} | "
            f"{'yes' if result.same_edge_set else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Both algorithms returned valid spanning trees with identical total "
            "weight in every case. Equal total weight is the correctness criterion; "
            "the selected edge sets may differ when several minimum spanning trees "
            "exist due to tied weights.",
            "",
            "The timings are a small local experiment, not a universal performance "
            "ranking. Kruskal pays for globally sorting all edges, while Prim pays "
            "for heap operations along the growing tree frontier. Graph density, "
            "input representation, constants, and runtime noise affect the result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_directory = Path(__file__).resolve().parent
    output_directory = project_directory / "results"
    results = run_benchmark()
    csv_path = output_directory / "mst_comparison.csv"
    report_path = output_directory / "REPORT.md"
    write_benchmark_csv(results, csv_path)
    write_report(results, report_path)

    print("Seeded MST workshop")
    print("vertices  edges  seed  weight  kruskal_us  prim_us  same_edges")
    for result in results:
        print(
            f"{result.vertex_count:>8}  {result.edge_count:>5}  "
            f"{result.seed:>4}  {result.mst_weight:>6.1f}  "
            f"{result.kruskal_median_us:>11.3f}  "
            f"{result.prim_median_us:>7.3f}  "
            f"{str(result.same_edge_set):>10}"
        )
    print(f"\nWrote {csv_path.relative_to(project_directory)}")
    print(f"Wrote {report_path.relative_to(project_directory)}")


if __name__ == "__main__":
    main()
