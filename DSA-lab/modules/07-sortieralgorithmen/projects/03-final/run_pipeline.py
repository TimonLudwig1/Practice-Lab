"""Run the reproducible external-sorting case study."""

from __future__ import annotations

import csv
from pathlib import Path

from external_sort import (
    SortMetrics,
    VerificationResult,
    external_sort_csv,
    verify_sorted_output,
)
from generate_data import DEFAULT_SEED, generate_events


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
INPUT_PATH = DATA_DIR / "unsorted_events.csv"
SORTED_PATH = OUTPUT_DIR / "sorted_events.csv"
METRICS_PATH = OUTPUT_DIR / "run_metrics.csv"
REPORT_PATH = OUTPUT_DIR / "RUN_REPORT.md"
RECORD_COUNT = 5_000
MEMORY_LIMIT_RECORDS = 250
MERGE_FAN_IN = 8
KEY_FIELDS = ("timestamp", "sensor_id")


def _write_metrics(
    path: Path, metrics: SortMetrics, verification: VerificationResult
) -> None:
    rows = (
        ("record_count", metrics.record_count),
        ("memory_limit_records", MEMORY_LIMIT_RECORDS),
        ("merge_fan_in", MERGE_FAN_IN),
        ("initial_runs", metrics.initial_runs),
        ("merge_passes", metrics.merge_passes),
        ("runs_by_pass", " -> ".join(map(str, metrics.runs_by_pass))),
        ("max_chunk_records", metrics.max_chunk_records),
        ("max_heap_entries", metrics.max_heap_entries),
        ("verification_ok", verification.ok),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def _write_report(
    path: Path, metrics: SortMetrics, verification: VerificationResult
) -> None:
    input_megabytes = INPUT_PATH.stat().st_size / 1_000_000
    output_megabytes = SORTED_PATH.stat().st_size / 1_000_000
    run_sequence = " → ".join(map(str, metrics.runs_by_pass))
    formatted_count = f"{metrics.record_count:,}".replace(",", ".")
    sorted_label = "ja" if verification.is_sorted else "nein"
    records_label = "ja" if verification.same_records else "nein"
    text = f"""# Laufbericht: Externe Sortierung

## Szenario

Die Pipeline sortiert {formatted_count} künstlich erzeugte Sensorereignisse
stabil nach `timestamp` und `sensor_id`. Der Generator arbeitet mit Seed
`{DEFAULT_SEED}`. Die CSV ist absichtlich nicht vorsortiert und enthält gleiche
Sortierschlüssel sowie Felder mit Kommata.

## Künstliche Ressourcengrenzen

- Höchstens **{MEMORY_LIMIT_RECORDS} Datensätze** dürfen gleichzeitig als Chunk
  sortiert werden.
- Ein Merge verbindet höchstens **{MERGE_FAN_IN} Runs** gleichzeitig.
- Die beobachteten Maxima lagen bei {metrics.max_chunk_records} Datensätzen pro
  Chunk und {metrics.max_heap_entries} Heap-Einträgen im k-Way-Merge.

## Ablauf und Ergebnis

- Eingabedatei: {input_megabytes:.2f} MB
- Ausgabedatei: {output_megabytes:.2f} MB
- Initiale sortierte Runs: {metrics.initial_runs}
- Runs je Stufe: {run_sequence}
- Merge-Pässe: {metrics.merge_passes}
- Sortierreihenfolge korrekt: {sorted_label}
- Datensätze vollständig und unverändert: {records_label}

Die versteckte ursprüngliche Zeilennummer ist der letzte Sortierschlüssel. Darum
bleibt die Reihenfolge gleicher öffentlicher Schlüssel auch über Chunk-Grenzen
hinweg stabil. Die Nummer wird nicht in die Ausgabe geschrieben.

## Einordnung

Das Verfahren ersetzt die unmögliche Annahme „alle Daten passen in den RAM“ durch
zwei beschränkte Schritte: lokal sortierte Runs und wiederholte k-Way-Merges. Für
`r` gleichzeitig geöffnete Runs enthält der Heap nur `r` Köpfe; jede Ausgabezeile
verursacht damit `O(log r)` Heap-Arbeit. Die I/O-Kosten wachsen mit der Anzahl der
Merge-Pässe. Ein größerer Fan-in spart Pässe, benötigt aber mehr Dateideskriptoren
und Heap-Einträge. Dieses Muster bildet die Grundlage externer Datenbank-Sorts
und verteilter Sortierphasen in Datenpipelines.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """Generate, sort, verify, and report the standard scenario."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_events(INPUT_PATH, record_count=RECORD_COUNT, seed=DEFAULT_SEED)
    metrics = external_sort_csv(
        INPUT_PATH,
        SORTED_PATH,
        key_fields=KEY_FIELDS,
        memory_limit_records=MEMORY_LIMIT_RECORDS,
        merge_fan_in=MERGE_FAN_IN,
    )
    verification = verify_sorted_output(
        INPUT_PATH, SORTED_PATH, key_fields=KEY_FIELDS
    )
    if not verification.ok:
        raise RuntimeError(f"output verification failed: {verification}")
    _write_metrics(METRICS_PATH, metrics, verification)
    _write_report(REPORT_PATH, metrics, verification)
    print(
        f"Sorted {metrics.record_count} records: "
        f"{' -> '.join(map(str, metrics.runs_by_pass))} runs, "
        f"verification={verification.ok}"
    )


if __name__ == "__main__":
    main()
