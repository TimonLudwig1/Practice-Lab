"""Tests der DAG-Invarianten, Graphanalysen und Berichtsausgabe."""

import csv
from pathlib import Path

import pytest

from pipeline_analysis import (
    BASE_EDGES,
    OPTIONAL_EDGES,
    TASK_SPECS,
    CriticalPath,
    PipelineCycleError,
    PipelineDAG,
    PipelineTask,
    generate_pipeline,
)
from reporting import write_analysis


def task(name: str, duration: int = 1) -> PipelineTask:
    return PipelineTask(name=name, duration_minutes=duration, category="test")


def diamond() -> PipelineDAG:
    return PipelineDAG(
        tasks=(task("A", 3), task("B", 5), task("C", 2), task("D", 7)),
        edges=(("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")),
    )


class TestConstruction:
    def test_empty_pipeline_is_valid(self) -> None:
        pipeline = PipelineDAG((), ())
        assert pipeline.task_count == 0
        assert pipeline.edge_count == 0
        assert pipeline.roots == ()
        assert pipeline.sinks == ()

    def test_duplicate_task_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="eindeutig"):
            PipelineDAG((task("A"), task("A")), ())

    def test_empty_task_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="nicht leer"):
            PipelineDAG((task(""),), ())

    @pytest.mark.parametrize("duration", [0, -1])
    def test_non_positive_duration_rejected(self, duration: int) -> None:
        with pytest.raises(ValueError, match="positiv"):
            PipelineDAG((task("A", duration),), ())

    @pytest.mark.parametrize("edge", [("X", "A"), ("A", "X")])
    def test_unknown_edge_endpoint_rejected(self, edge: tuple[str, str]) -> None:
        with pytest.raises(ValueError, match="unbekannten Task"):
            PipelineDAG((task("A"),), (edge,))

    def test_self_dependency_rejected(self) -> None:
        with pytest.raises(PipelineCycleError, match="Selbstabhängigkeit"):
            PipelineDAG((task("A"),), (("A", "A"),))

    def test_longer_cycle_rejected(self) -> None:
        with pytest.raises(PipelineCycleError, match="Zyklus"):
            PipelineDAG(
                (task("A"), task("B"), task("C")),
                (("A", "B"), ("B", "C"), ("C", "A")),
            )

    def test_duplicate_edge_is_deduplicated(self) -> None:
        pipeline = PipelineDAG((task("A"), task("B")), (("A", "B"), ("A", "B")))
        assert pipeline.edges == (("A", "B"),)
        assert pipeline.edge_count == 1

    def test_accessors(self) -> None:
        pipeline = diamond()
        assert pipeline.task("A") == task("A", 3)
        assert pipeline.prerequisites("D") == ("B", "C")
        assert pipeline.dependents("A") == ("B", "C")
        assert pipeline.roots == ("A",)
        assert pipeline.sinks == ("D",)

    @pytest.mark.parametrize("method", ["task", "prerequisites", "dependents", "upstream", "downstream", "failure_impact"])
    def test_unknown_task_rejected(self, method: str) -> None:
        with pytest.raises(KeyError, match="Unbekannter Pipeline-Task"):
            getattr(diamond(), method)("X")


class TestSeedGenerator:
    def test_expected_number_of_tasks(self) -> None:
        pipeline = generate_pipeline()
        assert pipeline.task_count == len(TASK_SPECS) == 24

    def test_edge_count_in_expected_range(self) -> None:
        pipeline = generate_pipeline()
        assert len(BASE_EDGES) <= pipeline.edge_count <= len(BASE_EDGES) + len(OPTIONAL_EDGES)

    def test_same_seed_is_identical(self) -> None:
        first = generate_pipeline(1203)
        second = generate_pipeline(1203)
        assert first.tasks == second.tasks
        assert first.edges == second.edges

    def test_different_seed_changes_details(self) -> None:
        first = generate_pipeline(1203)
        second = generate_pipeline(1204)
        assert first.tasks != second.tasks or first.edges != second.edges

    def test_generated_pipeline_is_dag(self) -> None:
        pipeline = generate_pipeline()
        order = pipeline.topological_order()
        assert pipeline.is_valid_order(order)

    def test_generated_pipeline_has_expected_sources_and_sink(self) -> None:
        pipeline = generate_pipeline()
        assert pipeline.roots == (
            "ingest_customers",
            "ingest_orders",
            "ingest_products",
        )
        assert pipeline.sinks == ("notify_complete",)

    def test_durations_within_seeded_range(self) -> None:
        assert all(2 <= item.duration_minutes <= 18 for item in generate_pipeline().tasks)


class TestTopologicalOrder:
    def test_diamond_order_is_valid(self) -> None:
        pipeline = diamond()
        assert pipeline.topological_order() == ("A", "B", "C", "D")
        assert pipeline.is_valid_order(("A", "C", "B", "D"))

    @pytest.mark.parametrize(
        "order",
        [
            ("B", "A", "C", "D"),
            ("A", "B", "C"),
            ("A", "B", "B", "D"),
            ("A", "B", "C", "X"),
        ],
    )
    def test_invalid_orders_rejected(self, order: tuple[str, ...]) -> None:
        assert not diamond().is_valid_order(order)

    def test_levels_capture_parallel_waves(self) -> None:
        assert diamond().levels() == {"A": 0, "B": 1, "C": 1, "D": 2}

    def test_earliest_schedule_uses_longer_parent(self) -> None:
        assert diamond().earliest_schedule() == {
            "A": (0, 3),
            "B": (3, 8),
            "C": (3, 5),
            "D": (8, 15),
        }

    def test_empty_critical_path(self) -> None:
        assert PipelineDAG((), ()).critical_path() == CriticalPath((), 0)

    def test_weighted_critical_path(self) -> None:
        assert diamond().critical_path() == CriticalPath(("A", "B", "D"), 15)


class TestReachabilityAndFailure:
    def test_downstream_bfs_order(self) -> None:
        assert diamond().downstream("A") == ("B", "C", "D")

    def test_upstream_reverse_bfs_order(self) -> None:
        assert diamond().upstream("D") == ("B", "C", "A")

    def test_sink_has_no_downstream_tasks(self) -> None:
        assert diamond().downstream("D") == ()

    def test_root_has_no_upstream_tasks(self) -> None:
        assert diamond().upstream("A") == ()

    def test_failure_impact_includes_failed_task_once(self) -> None:
        impact = diamond().failure_impact("A")
        assert impact.directly_blocked == ("B", "C")
        assert impact.transitively_blocked == ("B", "C", "D")
        assert impact.unavailable_tasks == ("A", "B", "C", "D")
        assert impact.blocked_count == 3
        assert impact.unavailable_count == 4
        assert impact.unaffected_count == 0

    def test_sink_failure_only_loses_sink(self) -> None:
        impact = diamond().failure_impact("D")
        assert impact.blocked_count == 0
        assert impact.unavailable_tasks == ("D",)
        assert impact.unaffected_count == 3

    def test_failure_ranking_uses_blast_radius(self) -> None:
        impacts = diamond().ranked_failure_impacts()
        assert tuple(item.failed_task for item in impacts) == ("A", "B", "C", "D")
        assert tuple(item.blocked_count for item in impacts) == (3, 1, 1, 0)

    def test_failure_ranking_tie_is_insertion_stable(self) -> None:
        pipeline = PipelineDAG((task("Z"), task("A")), ())
        assert tuple(item.failed_task for item in pipeline.ranked_failure_impacts()) == ("Z", "A")


class TestMetrics:
    def test_metric_count_and_order(self) -> None:
        pipeline = diamond()
        metrics = pipeline.task_metrics()
        assert len(metrics) == pipeline.task_count
        assert tuple(metric.task for metric in metrics) == pipeline.topological_order()

    def test_root_metrics(self) -> None:
        metric = diamond().task_metrics()[0]
        assert metric.task == "A"
        assert metric.topological_position == 0
        assert metric.level == 0
        assert metric.in_degree == 0
        assert metric.out_degree == 2
        assert metric.upstream_count == 0
        assert metric.downstream_count == 3
        assert metric.on_critical_path

    def test_shorter_branch_not_on_critical_path(self) -> None:
        metrics = {metric.task: metric for metric in diamond().task_metrics()}
        assert not metrics["C"].on_critical_path
        assert metrics["D"].earliest_start == 8
        assert metrics["D"].earliest_finish == 15


class TestReporting:
    def test_all_artifacts_created(self, tmp_path: Path) -> None:
        artifacts = write_analysis(tmp_path, seed=1203)
        assert artifacts.output_directory == tmp_path
        assert artifacts.task_metrics_csv.is_file()
        assert artifacts.edges_csv.is_file()
        assert artifacts.failure_impacts_csv.is_file()
        assert artifacts.report_markdown.is_file()

    def test_task_csv_has_one_row_per_task(self, tmp_path: Path) -> None:
        path = write_analysis(tmp_path).task_metrics_csv
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 24
        assert set(rows[0]) == {
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
        }

    def test_edge_csv_has_one_row_per_edge(self, tmp_path: Path) -> None:
        pipeline = generate_pipeline()
        path = write_analysis(tmp_path).edges_csv
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == pipeline.edge_count

    def test_failure_csv_is_ranked(self, tmp_path: Path) -> None:
        path = write_analysis(tmp_path).failure_impacts_csv
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        blocked = [int(row["blocked_descendants"]) for row in rows]
        assert len(rows) == 24
        assert blocked == sorted(blocked, reverse=True)
        assert [int(row["rank"]) for row in rows] == list(range(1, 25))

    def test_markdown_report_contains_core_sections(self, tmp_path: Path) -> None:
        report = write_analysis(tmp_path, seed=99).report_markdown.read_text(encoding="utf-8")
        assert "Seed: `99`" in report
        assert "DAG-Prüfung: **bestanden**" in report
        assert "## Gültige Ausführungsreihenfolge" in report
        assert "## Laufzeitkritischer Pfad" in report
        assert "## Kritische Knoten nach Ausfallreichweite" in report
        assert "## Interpretation" in report

    def test_repeated_write_is_reproducible(self, tmp_path: Path) -> None:
        first = write_analysis(tmp_path, seed=1203)
        before = {
            path.name: path.read_bytes()
            for path in (
                first.task_metrics_csv,
                first.edges_csv,
                first.failure_impacts_csv,
                first.report_markdown,
            )
        }
        second = write_analysis(tmp_path, seed=1203)
        after = {
            path.name: path.read_bytes()
            for path in (
                second.task_metrics_csv,
                second.edges_csv,
                second.failure_impacts_csv,
                second.report_markdown,
            )
        }
        assert before == after
