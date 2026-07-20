"""CSV- und Markdown-Ausgabe für die Pipeline-Abhängigkeitsanalyse."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from pipeline_analysis import PipelineDAG, generate_pipeline


@dataclass(frozen=True)
class AnalysisArtifacts:
    output_directory: Path
    task_metrics_csv: Path
    edges_csv: Path
    failure_impacts_csv: Path
    report_markdown: Path


def write_analysis(output_directory: Path, *, seed: int = 1203) -> AnalysisArtifacts:
    """Generiert den Seed-DAG und schreibt alle Analyseartefakte."""

    output_directory.mkdir(parents=True, exist_ok=True)
    pipeline = generate_pipeline(seed)

    artifacts = AnalysisArtifacts(
        output_directory=output_directory,
        task_metrics_csv=output_directory / "task_metrics.csv",
        edges_csv=output_directory / "pipeline_edges.csv",
        failure_impacts_csv=output_directory / "failure_impacts.csv",
        report_markdown=output_directory / "pipeline_report.md",
    )

    _write_task_metrics(pipeline, artifacts.task_metrics_csv)
    _write_edges(pipeline, artifacts.edges_csv)
    _write_failure_impacts(pipeline, artifacts.failure_impacts_csv)
    artifacts.report_markdown.write_text(
        _build_markdown_report(pipeline, seed=seed), encoding="utf-8"
    )
    return artifacts


def _write_task_metrics(pipeline: PipelineDAG, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "task",
                "category",
                "topological_position",
                "level",
                "duration_minutes",
                "earliest_start",
                "earliest_finish",
                "in_degree",
                "out_degree",
                "upstream_count",
                "downstream_count",
                "on_critical_path",
            )
        )
        for metric in pipeline.task_metrics():
            writer.writerow(
                (
                    metric.task,
                    pipeline.task(metric.task).category,
                    metric.topological_position,
                    metric.level,
                    metric.duration_minutes,
                    metric.earliest_start,
                    metric.earliest_finish,
                    metric.in_degree,
                    metric.out_degree,
                    metric.upstream_count,
                    metric.downstream_count,
                    str(metric.on_critical_path).lower(),
                )
            )


def _write_edges(pipeline: PipelineDAG, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("prerequisite", "dependent"))
        writer.writerows(pipeline.edges)


def _write_failure_impacts(pipeline: PipelineDAG, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "rank",
                "failed_task",
                "directly_blocked_count",
                "blocked_descendants",
                "unavailable_tasks",
                "unavailable_percent",
                "unaffected_count",
                "directly_blocked",
                "transitively_blocked",
            )
        )
        for rank, impact in enumerate(pipeline.ranked_failure_impacts(), start=1):
            writer.writerow(
                (
                    rank,
                    impact.failed_task,
                    len(impact.directly_blocked),
                    impact.blocked_count,
                    impact.unavailable_count,
                    f"{100 * impact.unavailable_count / pipeline.task_count:.2f}",
                    impact.unaffected_count,
                    "|".join(impact.directly_blocked),
                    "|".join(impact.transitively_blocked),
                )
            )


def _build_markdown_report(pipeline: PipelineDAG, *, seed: int) -> str:
    order = pipeline.topological_order()
    levels = pipeline.levels()
    critical_path = pipeline.critical_path()
    impacts = pipeline.ranked_failure_impacts()

    waves: dict[int, list[str]] = {}
    for task in order:
        waves.setdefault(levels[task], []).append(task)

    lines = [
        "# Abhängigkeitsanalyse der synthetischen Datenpipeline",
        "",
        "## Überblick",
        "",
        f"- Seed: `{seed}`",
        f"- Tasks: **{pipeline.task_count}**",
        f"- Abhängigkeiten: **{pipeline.edge_count}**",
        f"- Quellen: {', '.join(f'`{task}`' for task in pipeline.roots)}",
        f"- Senken: {', '.join(f'`{task}`' for task in pipeline.sinks)}",
        "- DAG-Prüfung: **bestanden** (Kahn verarbeitet alle Tasks)",
        "",
        "## Gültige Ausführungsreihenfolge",
        "",
    ]
    lines.extend(f"{index}. `{task}`" for index, task in enumerate(order, start=1))

    lines.extend(
        [
            "",
            "Diese Reihenfolge ist eine von möglicherweise mehreren gültigen "
            "Toposorts. Unabhängige Tasks dürfen parallel laufen.",
            "",
            "## Frühestmögliche Ausführungswellen",
            "",
            "| Welle | Tasks |",
            "|---:|---|",
        ]
    )
    lines.extend(
        f"| {level} | {', '.join(f'`{task}`' for task in tasks)} |"
        for level, tasks in waves.items()
    )

    lines.extend(
        [
            "",
            "## Laufzeitkritischer Pfad",
            "",
            " → ".join(f"`{task}`" for task in critical_path.tasks),
            "",
            f"Gesamtdauer bei unbegrenzter Parallelität: "
            f"**{critical_path.duration_minutes} Minuten**.",
            "",
            "Ein Task auf diesem Pfad verzögert bei eigener Verzögerung den "
            "frühestmöglichen Pipeline-Abschluss, sofern kein anderer Pfad "
            "gleich lang wird.",
            "",
            "## Kritische Knoten nach Ausfallreichweite",
            "",
            "Hier bedeutet *kritisch*: Ein Ausfall blockiert viele transitive "
            "Nachfolger. Der ausgefallene Task selbst wird separat als nicht "
            "verfügbar gezählt.",
            "",
            "| Rang | Task | direkte Blockaden | blockierte Nachfolger | "
            "nicht verfügbar | Anteil |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for rank, impact in enumerate(impacts[:8], start=1):
        percentage = 100 * impact.unavailable_count / pipeline.task_count
        lines.append(
            f"| {rank} | `{impact.failed_task}` | "
            f"{len(impact.directly_blocked)} | {impact.blocked_count} | "
            f"{impact.unavailable_count} | {percentage:.1f} % |"
        )

    most_critical = impacts[0] if impacts else None
    lines.extend(["", "## Interpretation", ""])
    if most_critical is not None:
        lines.append(
            f"Der größte einzelne Blast Radius entsteht bei "
            f"`{most_critical.failed_task}`: Mit diesem Task sind "
            f"{most_critical.unavailable_count} von {pipeline.task_count} Tasks "
            f"nicht verfügbar."
        )
    lines.extend(
        [
            "",
            "Die Ausfallanalyse ist reine Erreichbarkeit: Von einem ausgefallenen "
            "Knoten aus werden alle Nachfolger per BFS markiert. Das entspricht "
            "dem Grundmodell eines Schedulers wie Airflow: Ein Task kann nur "
            "laufen, wenn sämtliche Voraussetzungen erfolgreich waren.",
            "",
            "Der Blast-Radius-Rang und der laufzeitkritische Pfad beantworten "
            "verschiedene Fragen. Ein früher Ingest-Task kann viele Nachfolger "
            "blockieren, ohne auf dem längsten Laufzeitpfad zu liegen; ein langer "
            "Modell-Task kann den Abschluss bestimmen, obwohl er weniger "
            "Nachfolger besitzt.",
            "",
        ]
    )
    return "\n".join(lines)
