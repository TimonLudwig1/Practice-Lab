"""Erzeugt Pipeline, CSV-Auswertungen und den Markdown-Bericht."""

from pathlib import Path

from pipeline_analysis import generate_pipeline
from reporting import write_analysis


SEED = 1203


def main() -> None:
    output_directory = Path(__file__).resolve().parent / "results"
    artifacts = write_analysis(output_directory, seed=SEED)
    pipeline = generate_pipeline(SEED)
    path = pipeline.critical_path()
    most_critical = pipeline.ranked_failure_impacts()[0]

    print(f"Seed: {SEED}")
    print(f"DAG: {pipeline.task_count} Tasks, {pipeline.edge_count} Kanten")
    print(f"Quellen: {', '.join(pipeline.roots)}")
    print(f"Senken: {', '.join(pipeline.sinks)}")
    print(f"Toposort gültig: {pipeline.is_valid_order(pipeline.topological_order())}")
    print(
        "Kritischer Pfad: "
        + " -> ".join(path.tasks)
        + f" ({path.duration_minutes} Minuten)"
    )
    print(
        f"Größter Blast Radius: {most_critical.failed_task} blockiert "
        f"{most_critical.blocked_count} Nachfolger"
    )
    print("\nErzeugte Dateien:")
    for generated in (
        artifacts.task_metrics_csv,
        artifacts.edges_csv,
        artifacts.failure_impacts_csv,
        artifacts.report_markdown,
    ):
        print(f"  {generated.relative_to(Path(__file__).resolve().parent)}")


if __name__ == "__main__":
    main()
