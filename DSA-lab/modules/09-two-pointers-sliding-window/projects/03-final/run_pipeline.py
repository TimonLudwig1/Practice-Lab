"""Run and report the complete streaming anomaly-detection case study."""

from __future__ import annotations

import csv
import os
import statistics
import tempfile
import time
from pathlib import Path

from anomaly_detector import (
    Detection,
    detect_naive,
    detect_streaming,
    evaluate_detections,
    read_metric_csv,
)
from generate_stream import DEFAULT_SEED, generate_metric_stream


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_PATH = DATA_DIR / "metric_stream.csv"
DETECTIONS_PATH = OUTPUT_DIR / "detections.csv"
METRICS_PATH = OUTPUT_DIR / "benchmark_metrics.csv"
PLOT_PATH = OUTPUT_DIR / "anomaly_detection.png"
REPORT_PATH = OUTPUT_DIR / "RUN_REPORT.md"
RECORD_COUNT = 12_000
ANOMALY_COUNT = 20
WINDOW_SIZE = 120
Z_THRESHOLD = 5.0
REPETITIONS = 3


def _median_runtime(operation) -> tuple[float, list[Detection]]:
    samples = []
    result = []
    for _ in range(REPETITIONS):
        start = time.perf_counter_ns()
        result = operation()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(samples), result


def _equivalent(left: list[Detection], right: list[Detection]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        first.index == second.index
        and first.is_anomaly == second.is_anomaly
        and first.injected_anomaly == second.injected_anomaly
        and abs(first.rolling_mean - second.rolling_mean) < 1e-9
        and abs(first.rolling_std - second.rolling_std) < 1e-8
        for first, second in zip(left, right)
    )


def _write_detections(detections: list[Detection]) -> None:
    with DETECTIONS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "index",
                "timestamp",
                "value",
                "rolling_mean",
                "rolling_std",
                "z_score",
                "is_anomaly",
                "injected_anomaly",
            )
        )
        for item in detections:
            writer.writerow(
                (
                    item.index,
                    item.timestamp,
                    f"{item.value:.8f}",
                    f"{item.rolling_mean:.8f}",
                    f"{item.rolling_std:.8f}",
                    f"{item.z_score:.8f}",
                    item.is_anomaly,
                    item.injected_anomaly,
                )
            )


def _write_metrics(naive_ms: float, streaming_ms: float, metrics) -> None:
    with METRICS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerow(("record_count", RECORD_COUNT))
        writer.writerow(("window_size", WINDOW_SIZE))
        writer.writerow(("z_threshold", Z_THRESHOLD))
        writer.writerow(("naive_median_ms", f"{naive_ms:.6f}"))
        writer.writerow(("streaming_median_ms", f"{streaming_ms:.6f}"))
        writer.writerow(("speedup", f"{naive_ms / streaming_ms:.3f}"))
        writer.writerow(("true_positives", metrics.true_positives))
        writer.writerow(("false_positives", metrics.false_positives))
        writer.writerow(("false_negatives", metrics.false_negatives))
        writer.writerow(("precision", f"{metrics.precision:.6f}"))
        writer.writerow(("recall", f"{metrics.recall:.6f}"))


def _write_plot(detections: list[Detection]) -> None:
    cache = Path(tempfile.gettempdir()) / "dsa-lab-plot-cache-module-09"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    indices = [item.index for item in detections]
    values = [item.value for item in detections]
    means = [item.rolling_mean for item in detections]
    anomalies = [item for item in detections if item.is_anomaly]
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    axis.plot(indices, values, linewidth=0.7, alpha=0.65, label="Messwert")
    axis.plot(indices, means, linewidth=1.5, label="Rolling Mean")
    axis.scatter(
        [item.index for item in anomalies],
        [item.value for item in anomalies],
        color="tab:red",
        marker="x",
        s=45,
        label="erkannte Anomalie",
        zorder=3,
    )
    axis.set(
        title="Streaming-Anomalieerkennung mit vorherigem Rolling Window",
        xlabel="Streamindex",
        ylabel="Metrikwert",
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.savefig(PLOT_PATH, dpi=160)
    plt.close(figure)


def _write_report(naive_ms: float, streaming_ms: float, metrics) -> None:
    speedup = naive_ms / streaming_ms
    text = f"""# Laufbericht: Streaming-Anomalieerkennung

## Szenario

Der Generator erzeugt mit Seed `{DEFAULT_SEED}` insgesamt {RECORD_COUNT:,}
Minutenmesswerte mit langsamem periodischem Verlauf, Gaußrauschen und
{ANOMALY_COUNT} weit auseinanderliegenden injizierten Ausreißern.

Jeder neue Wert wird ausschließlich gegen die **vorherigen {WINDOW_SIZE} Werte**
bewertet. Damit gibt es weder Look-ahead noch eine Aufnahme des zu prüfenden
Werts in seine eigene Baseline. Eine Anomalie liegt bei
`|z| > {Z_THRESHOLD:.1f}` vor.

## Ergebnis

- True Positives: {metrics.true_positives}
- False Positives: {metrics.false_positives}
- False Negatives: {metrics.false_negatives}
- Precision: {metrics.precision:.4f}
- Recall: {metrics.recall:.4f}

## Laufzeitvergleich

- Naive Neuberechnung, Median: {naive_ms:.3f} ms
- O(1)-Rolling-Update, Median: {streaming_ms:.3f} ms
- Beschleunigung: {speedup:.2f}x

Beide Verfahren erzeugen dieselben Anomalieflags sowie innerhalb enger Toleranz
dieselben Mittelwerte und Standardabweichungen. Die Referenz summiert für jeden
Punkt ein Fenster der Breite `k` neu und kostet `O(nk)`. Die Streaming-Variante
entfernt einen Wert aus Summe und Quadratsumme und fügt einen Wert hinzu. Damit
kostet jeder Schritt `O(1)` und der gesamte Lauf `O(n)`.

Die Varianzformel `E[x²] - E[x]²` ist schnell, kann bei sehr großen fast gleichen
Zahlen aber Auslöschung zeigen. Die Implementierung klemmt winzige negative
Rundungsreste auf null; für numerisch extreme Produktionsdaten wären stabilere
Online-Verfahren oder periodische Rekalibrierung zu prüfen.
""".replace(f"{RECORD_COUNT:,}", f"{RECORD_COUNT:,}".replace(",", "."))
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    """Generate, detect, compare, and persist the standard scenario."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_metric_stream(
        DATA_PATH,
        record_count=RECORD_COUNT,
        anomaly_count=ANOMALY_COUNT,
        seed=DEFAULT_SEED,
    )
    points = read_metric_csv(DATA_PATH)
    naive_ms, naive = _median_runtime(
        lambda: detect_naive(
            points, window_size=WINDOW_SIZE, z_threshold=Z_THRESHOLD
        )
    )
    streaming_ms, streaming = _median_runtime(
        lambda: detect_streaming(
            points, window_size=WINDOW_SIZE, z_threshold=Z_THRESHOLD
        )
    )
    if not _equivalent(naive, streaming):
        raise RuntimeError("streaming output differs from naive reference")
    metrics = evaluate_detections(streaming)
    _write_detections(streaming)
    _write_metrics(naive_ms, streaming_ms, metrics)
    _write_plot(streaming)
    _write_report(naive_ms, streaming_ms, metrics)
    print(
        f"Processed {len(points)} points: {metrics.true_positives} TP, "
        f"{metrics.false_positives} FP, {metrics.false_negatives} FN, "
        f"speedup={naive_ms / streaming_ms:.2f}x"
    )


if __name__ == "__main__":
    main()
