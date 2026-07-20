"""Reproduzierbare Abhängigkeitsanalyse für einen synthetischen Pipeline-DAG."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random


class PipelineCycleError(ValueError):
    """Die Abhängigkeiten enthalten einen Zyklus und bilden keinen DAG."""


@dataclass(frozen=True)
class PipelineTask:
    """Ein Pipeline-Task mit simulierter serieller Laufzeit."""

    name: str
    duration_minutes: int
    category: str


@dataclass(frozen=True)
class CriticalPath:
    """Längster laufzeitgewichteter Pfad durch den DAG."""

    tasks: tuple[str, ...]
    duration_minutes: int


@dataclass(frozen=True)
class FailureImpact:
    """Direkte und transitive Auswirkungen eines einzelnen Task-Ausfalls."""

    failed_task: str
    directly_blocked: tuple[str, ...]
    transitively_blocked: tuple[str, ...]
    unavailable_tasks: tuple[str, ...]
    unaffected_count: int

    @property
    def blocked_count(self) -> int:
        return len(self.transitively_blocked)

    @property
    def unavailable_count(self) -> int:
        return len(self.unavailable_tasks)


@dataclass(frozen=True)
class TaskMetrics:
    """Graphmetriken und Scheduling-Metadaten eines Tasks."""

    task: str
    topological_position: int
    level: int
    duration_minutes: int
    earliest_start: int
    earliest_finish: int
    in_degree: int
    out_degree: int
    upstream_count: int
    downstream_count: int
    on_critical_path: bool


class PipelineDAG:
    """Gerichteter azyklischer Graph aus Tasks und Abhängigkeitskanten.

    Eine Kante ``(A, B)`` bedeutet: B darf erst nach A starten. Alle öffentlichen
    Reihenfolgen sind stabil und folgen bei Gleichstand der Task-Einfügung.
    """

    def __init__(
        self,
        tasks: tuple[PipelineTask, ...],
        edges: tuple[tuple[str, str], ...],
    ) -> None:
        if len({task.name for task in tasks}) != len(tasks):
            raise ValueError("Task-Namen müssen eindeutig sein")
        if any(not task.name for task in tasks):
            raise ValueError("Task-Namen dürfen nicht leer sein")
        if any(task.duration_minutes <= 0 for task in tasks):
            raise ValueError("Task-Laufzeiten müssen positiv sein")

        self._tasks = {task.name: task for task in tasks}
        self._outgoing: dict[str, dict[str, None]] = {
            task.name: {} for task in tasks
        }
        self._incoming: dict[str, dict[str, None]] = {
            task.name: {} for task in tasks
        }

        for prerequisite, dependent in edges:
            if prerequisite not in self._tasks or dependent not in self._tasks:
                raise ValueError(
                    f"Kante verweist auf unbekannten Task: "
                    f"{prerequisite!r}->{dependent!r}"
                )
            if prerequisite == dependent:
                raise PipelineCycleError(f"Selbstabhängigkeit bei {prerequisite!r}")
            self._outgoing[prerequisite].setdefault(dependent, None)
            self._incoming[dependent].setdefault(prerequisite, None)

        # Der Konstruktor garantiert die zentrale Klasseninvariante: DAG.
        self.topological_order()

    @property
    def tasks(self) -> tuple[PipelineTask, ...]:
        return tuple(self._tasks.values())

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (source, target)
            for source, targets in self._outgoing.items()
            for target in targets
        )

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    @property
    def edge_count(self) -> int:
        return sum(len(targets) for targets in self._outgoing.values())

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(task for task in self._tasks if not self._incoming[task])

    @property
    def sinks(self) -> tuple[str, ...]:
        return tuple(task for task in self._tasks if not self._outgoing[task])

    def task(self, name: str) -> PipelineTask:
        self._require_task(name)
        return self._tasks[name]

    def prerequisites(self, name: str) -> tuple[str, ...]:
        self._require_task(name)
        return tuple(self._incoming[name])

    def dependents(self, name: str) -> tuple[str, ...]:
        self._require_task(name)
        return tuple(self._outgoing[name])

    def topological_order(self) -> tuple[str, ...]:
        """Berechnet eine stabile gültige Ausführungsreihenfolge nach Kahn."""

        in_degree = {
            task: len(prerequisites)
            for task, prerequisites in self._incoming.items()
        }
        queue = deque(task for task in self._tasks if in_degree[task] == 0)
        order: list[str] = []

        while queue:
            task = queue.popleft()
            order.append(task)
            for dependent in self._outgoing[task]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != self.task_count:
            cyclic = tuple(task for task in self._tasks if in_degree[task] > 0)
            raise PipelineCycleError(
                f"Pipeline enthält einen Zyklus; Restknoten: {cyclic!r}"
            )
        return tuple(order)

    def is_valid_order(self, order: tuple[str, ...]) -> bool:
        if len(order) != self.task_count or len(set(order)) != len(order):
            return False
        if set(order) != set(self._tasks):
            return False
        position = {task: index for index, task in enumerate(order)}
        return all(
            position[source] < position[target] for source, target in self.edges
        )

    def downstream(self, name: str) -> tuple[str, ...]:
        """Ermittelt alle transitiv abhängigen Tasks per BFS."""

        self._require_task(name)
        visited = {name}
        queue = deque([name])
        result: list[str] = []

        while queue:
            current = queue.popleft()
            for dependent in self._outgoing[current]:
                if dependent not in visited:
                    visited.add(dependent)
                    queue.append(dependent)
                    result.append(dependent)
        return tuple(result)

    def upstream(self, name: str) -> tuple[str, ...]:
        """Ermittelt alle transitiven Voraussetzungen per rückwärts gerichteter BFS."""

        self._require_task(name)
        visited = {name}
        queue = deque([name])
        result: list[str] = []

        while queue:
            current = queue.popleft()
            for prerequisite in self._incoming[current]:
                if prerequisite not in visited:
                    visited.add(prerequisite)
                    queue.append(prerequisite)
                    result.append(prerequisite)
        return tuple(result)

    def levels(self) -> dict[str, int]:
        """Ordnet Tasks frühestmöglichen parallelen Ausführungswellen zu."""

        levels: dict[str, int] = {}
        for task in self.topological_order():
            levels[task] = (
                0
                if not self._incoming[task]
                else 1 + max(levels[parent] for parent in self._incoming[task])
            )
        return levels

    def earliest_schedule(self) -> dict[str, tuple[int, int]]:
        """Berechnet frühesten Start und Abschluss bei unbegrenzter Parallelität."""

        schedule: dict[str, tuple[int, int]] = {}
        for task in self.topological_order():
            start = max(
                (schedule[parent][1] for parent in self._incoming[task]),
                default=0,
            )
            finish = start + self._tasks[task].duration_minutes
            schedule[task] = (start, finish)
        return schedule

    def critical_path(self) -> CriticalPath:
        """Bestimmt den längsten laufzeitgewichteten Pfad per DAG-DP."""

        if not self._tasks:
            return CriticalPath(tasks=(), duration_minutes=0)

        finish: dict[str, int] = {}
        predecessor: dict[str, str | None] = {}
        for task in self.topological_order():
            parents = tuple(self._incoming[task])
            if parents:
                best_parent = max(parents, key=lambda parent: finish[parent])
                start = finish[best_parent]
                predecessor[task] = best_parent
            else:
                start = 0
                predecessor[task] = None
            finish[task] = start + self._tasks[task].duration_minutes

        end = max(self._tasks, key=lambda task: finish[task])
        reversed_path: list[str] = []
        current: str | None = end
        while current is not None:
            reversed_path.append(current)
            current = predecessor[current]
        reversed_path.reverse()
        return CriticalPath(tuple(reversed_path), finish[end])

    def failure_impact(self, failed_task: str) -> FailureImpact:
        """Bestimmt alle durch einen Ausfall blockierten Nachfolger."""

        self._require_task(failed_task)
        direct = self.dependents(failed_task)
        blocked = self.downstream(failed_task)
        unavailable = (failed_task, *blocked)
        return FailureImpact(
            failed_task=failed_task,
            directly_blocked=direct,
            transitively_blocked=blocked,
            unavailable_tasks=unavailable,
            unaffected_count=self.task_count - len(unavailable),
        )

    def ranked_failure_impacts(self) -> tuple[FailureImpact, ...]:
        """Sortiert Ausfälle nach Blast Radius, dann stabil nach Task-Reihenfolge."""

        insertion_position = {
            task: index for index, task in enumerate(self._tasks)
        }
        impacts = [self.failure_impact(task) for task in self._tasks]
        impacts.sort(
            key=lambda impact: (
                -impact.blocked_count,
                insertion_position[impact.failed_task],
            )
        )
        return tuple(impacts)

    def task_metrics(self) -> tuple[TaskMetrics, ...]:
        order = self.topological_order()
        position = {task: index for index, task in enumerate(order)}
        levels = self.levels()
        schedule = self.earliest_schedule()
        critical = set(self.critical_path().tasks)
        return tuple(
            TaskMetrics(
                task=task,
                topological_position=position[task],
                level=levels[task],
                duration_minutes=self._tasks[task].duration_minutes,
                earliest_start=schedule[task][0],
                earliest_finish=schedule[task][1],
                in_degree=len(self._incoming[task]),
                out_degree=len(self._outgoing[task]),
                upstream_count=len(self.upstream(task)),
                downstream_count=len(self.downstream(task)),
                on_critical_path=task in critical,
            )
            for task in order
        )

    def _require_task(self, name: str) -> None:
        if name not in self._tasks:
            raise KeyError(f"Unbekannter Pipeline-Task: {name!r}")


TASK_SPECS: tuple[tuple[str, str], ...] = (
    ("ingest_customers", "ingest"),
    ("ingest_orders", "ingest"),
    ("ingest_products", "ingest"),
    ("validate_customers", "validation"),
    ("validate_orders", "validation"),
    ("validate_products", "validation"),
    ("clean_customers", "transform"),
    ("clean_orders", "transform"),
    ("clean_products", "transform"),
    ("join_sales", "transform"),
    ("aggregate_daily", "transform"),
    ("build_customer_features", "feature"),
    ("build_product_features", "feature"),
    ("train_churn_model", "model"),
    ("train_demand_model", "model"),
    ("score_customers", "inference"),
    ("forecast_demand", "inference"),
    ("quality_sales", "quality"),
    ("quality_models", "quality"),
    ("publish_dashboard", "publish"),
    ("publish_scores", "publish"),
    ("publish_forecast", "publish"),
    ("archive_raw", "archive"),
    ("notify_complete", "notify"),
)


BASE_EDGES: tuple[tuple[str, str], ...] = (
    ("ingest_customers", "validate_customers"),
    ("ingest_orders", "validate_orders"),
    ("ingest_products", "validate_products"),
    ("ingest_customers", "archive_raw"),
    ("ingest_orders", "archive_raw"),
    ("ingest_products", "archive_raw"),
    ("validate_customers", "clean_customers"),
    ("validate_orders", "clean_orders"),
    ("validate_products", "clean_products"),
    ("clean_customers", "join_sales"),
    ("clean_orders", "join_sales"),
    ("clean_products", "join_sales"),
    ("join_sales", "aggregate_daily"),
    ("clean_customers", "build_customer_features"),
    ("aggregate_daily", "build_customer_features"),
    ("clean_products", "build_product_features"),
    ("aggregate_daily", "build_product_features"),
    ("build_customer_features", "train_churn_model"),
    ("build_product_features", "train_demand_model"),
    ("train_churn_model", "score_customers"),
    ("train_demand_model", "forecast_demand"),
    ("aggregate_daily", "quality_sales"),
    ("train_churn_model", "quality_models"),
    ("train_demand_model", "quality_models"),
    ("aggregate_daily", "publish_dashboard"),
    ("quality_sales", "publish_dashboard"),
    ("score_customers", "publish_scores"),
    ("quality_models", "publish_scores"),
    ("forecast_demand", "publish_forecast"),
    ("quality_models", "publish_forecast"),
    ("publish_dashboard", "notify_complete"),
    ("publish_scores", "notify_complete"),
    ("publish_forecast", "notify_complete"),
    ("archive_raw", "notify_complete"),
)


OPTIONAL_EDGES: tuple[tuple[str, str], ...] = (
    ("validate_orders", "quality_sales"),
    ("clean_orders", "build_customer_features"),
    ("clean_products", "quality_sales"),
    ("aggregate_daily", "train_demand_model"),
    ("build_customer_features", "quality_models"),
    ("build_product_features", "quality_models"),
    ("quality_models", "publish_dashboard"),
    ("archive_raw", "publish_dashboard"),
)


def generate_pipeline(seed: int = 1203) -> PipelineDAG:
    """Erzeugt denselben semantischen DAG mit Seed-basierten Details."""

    random = Random(seed)
    tasks = tuple(
        PipelineTask(
            name=name,
            duration_minutes=random.randint(2, 18),
            category=category,
        )
        for name, category in TASK_SPECS
    )
    optional = tuple(edge for edge in OPTIONAL_EDGES if random.random() < 0.6)
    return PipelineDAG(tasks=tasks, edges=BASE_EDGES + optional)
