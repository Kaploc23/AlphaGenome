#!/usr/bin/env python3
"""Interactive interval-centered saturation mutagenesis runner.

This script accepts one or more genomic intervals (e.g., enhancer regions),
runs AlphaGenome saturation mutagenesis for each interval, and writes:
- scored CSV
- raw pipeline plot
- styled plot (same visual style as existing workflow)
- run summary

It uses the interval midpoint as the reference coordinate for plotting and
for window construction, while scoring over the full provided interval.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import re
import subprocess
import sys
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import pandas as pd

MUTATION_COLORS = {
    "A": "#2ca25f",
    "T": "#2b8cbe",
    "C": "#de2d26",
    "G": "#ff7f0e",
}

X_OFFSETS = {
    "A": -0.27,
    "T": -0.09,
    "C": 0.09,
    "G": 0.27,
}

INTERVAL_RE = re.compile(r"^(?P<chrom>[^:]+):(?P<start>[0-9,]+)-(?P<end>[0-9,]+)$")
SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
INTERVAL_SEARCH_RE = re.compile(r"(?P<chrom>chr[0-9XYMxy]+):(?P<start>[0-9,]+)-(?P<end>[0-9,]+)")
CHROM_TOKEN_RE = re.compile(r"^(chr)?([0-9]{1,2}|X|Y|M|MT)$", re.IGNORECASE)

DATASET_PRESETS: list[tuple[str, str, list[str]]] = [
    (
        "general_cardiac",
        "General cardiac (heart + cardiac muscle cell)",
        ["UBERON:0000948", "CL:0000746"],
    ),
    (
        "atrial",
        "Atrial (cardiac atrium)",
        ["UBERON:0002079"],
    ),
    (
        "ventricular",
        "Ventricular (cardiac ventricle)",
        ["UBERON:0002084"],
    ),
]

OUTPUT_PRESETS: list[tuple[str, str, list[str]]] = [
    ("promoter", "Promoter-focused (CAGE)", ["CAGE"]),
    ("enhancer_basic", "Enhancer-focused basic (ATAC, DNASE, CAGE)", ["ATAC", "DNASE", "CAGE"]),
    ("enhancer_extended", "Enhancer-focused extended (ATAC, DNASE, CAGE, CHIP_HISTONE, CHIP_TF, PROCAP)", ["ATAC", "DNASE", "CAGE", "CHIP_HISTONE", "CHIP_TF", "PROCAP"]),
    ("tf_binding", "TF-binding focused (CHIP_TF, ATAC, DNASE)", ["CHIP_TF", "ATAC", "DNASE"]),
    ("transcript_abundance", "General transcript abundance (RNA_SEQ, CAGE)", ["RNA_SEQ", "CAGE"]),
    ("polyadenylation_shifts", "3' end processing / polyadenylation (PolyA / 3'-seq)", ["RNA_SEQ", "CAGE"]),
    ("translation_binding", "Translation and binding (Ribo-seq / eCLIP)", ["RNA_SEQ", "CAGE"]),
]

DATASET_POSTFILTER_KEYWORDS: dict[str, list[str]] = {
    "general_cardiac": ["heart", "cardiac"],
    "atrial": ["atrium", "atrial"],
    "ventricular": ["ventricle", "ventricular"],
}


@dataclass
class GenomicInterval:
    chrom: str
    start: int
    end: int


@dataclass
class IntervalTask:
    label: str
    interval: GenomicInterval


def parse_interval(text: str) -> GenomicInterval:
    raw = text.strip()
    match = INTERVAL_RE.match(raw)
    if not match:
        raise ValueError(f"Could not parse interval: {text}")
    chrom = match.group("chrom")
    start = int(match.group("start").replace(",", ""))
    end = int(match.group("end").replace(",", ""))
    if start < 1 or end < 1:
        raise ValueError("Interval positions must be >= 1")
    return GenomicInterval(chrom=chrom, start=min(start, end), end=max(start, end))


def prompt_nonempty(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Please enter a value.")


def prompt_positive_int(prompt: str) -> int:
    while True:
        raw = input(prompt).strip().replace(",", "")
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if value <= 0:
            print("Please enter a positive integer.")
            continue
        return value


def prompt_int_range(prompt: str, minimum: int, maximum: int) -> int:
    while True:
        value = prompt_positive_int(prompt)
        if value < minimum or value > maximum:
            print(f"Please enter a value between {minimum} and {maximum}.")
            continue
        return value


def prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    while True:
        raw = input(prompt + suffix).strip().lower()
        if not raw:
            return default_yes
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer y or n.")


def prompt_dataset_choice() -> tuple[str, str, list[str] | None, list[str] | None]:
    print("\nChoose dataset filter:")
    for index, (_, description, terms) in enumerate(DATASET_PRESETS, start=1):
        print(f"  {index}. {description} -> {', '.join(terms)}")
    print(f"  {len(DATASET_PRESETS) + 1}. All tissues (no ontology filter)")
    print(f"  {len(DATASET_PRESETS) + 2}. Custom ontology terms (any CURIEs)")

    minimum = 1
    maximum = len(DATASET_PRESETS) + 2
    choice = prompt_int_range(
        f"Select dataset option ({minimum}-{maximum}): ",
        minimum=minimum,
        maximum=maximum,
    )

    if choice <= len(DATASET_PRESETS):
        key, description, terms = DATASET_PRESETS[choice - 1]
        return key, f"{key}: {description}", list(terms), list(DATASET_POSTFILTER_KEYWORDS.get(key, []))

    if choice == len(DATASET_PRESETS) + 1:
        return "all_tissues", "all_tissues: no ontology filter", None, None

    custom_raw = prompt_nonempty("Enter ontology terms (comma-separated, any CURIEs, e.g., UBERON:0002079,CL:0000746): ")
    terms = [term.strip() for term in custom_raw.split(",") if term.strip()]
    if not terms:
        raise RuntimeError("No ontology terms provided for custom dataset")
    return "custom", "custom", terms, None


def prompt_output_choice() -> tuple[str, list[str], str | None]:
    print("\nChoose output preset:")
    for index, (key, description, outputs) in enumerate(OUTPUT_PRESETS, start=1):
        print(f"  {index}. {description} -> {', '.join(outputs)}")
    print(f"  {len(OUTPUT_PRESETS) + 1}. Custom output types")

    choice = prompt_int_range(
        f"Select output option (1-{len(OUTPUT_PRESETS) + 1}): ",
        minimum=1,
        maximum=len(OUTPUT_PRESETS) + 1,
    )

    if choice <= len(OUTPUT_PRESETS):
        key, description, outputs = OUTPUT_PRESETS[choice - 1]
        filter_key = None
        if key in {"transcript_abundance", "polyadenylation_shifts", "translation_binding"}:
            filter_key = key
        return f"{key}: {description}", list(outputs), filter_key

    custom_raw = prompt_nonempty(
        "Enter output types (comma-separated, e.g., ATAC,DNASE,CAGE,CHIP_TF): "
    )
    outputs = [value.strip().upper() for value in custom_raw.split(",") if value.strip()]
    if not outputs:
        raise RuntimeError("No output types provided for custom output selection")
    return "custom", outputs, None


def prompt_output_filter() -> str | None:
    print("\nOptional: choose an output filter preset to bias returned tracks:")
    filters = [
        ("transcript_abundance", "Transcript abundance (RNA-seq + CAGE)"),
        ("polyadenylation_shifts", "3' end processing / polyadenylation shifts (PolyA / 3'-seq)"),
        ("translation_binding", "Translation and binding (Ribo-seq / eCLIP)"),
    ]
    for i, (_, desc) in enumerate(filters, start=1):
        print(f"  {i}. {desc}")
    print(f"  {len(filters) + 1}. None (no output filter)")

    choice = prompt_int_range(
        f"Select output filter (1-{len(filters) + 1}): ",
        minimum=1,
        maximum=len(filters) + 1,
    )
    if choice <= len(filters):
        key, desc = filters[choice - 1]
        return key
    return None


def slugify(text: str) -> str:
    return SLUG_RE.sub("_", text.strip()).strip("_") or "interval"


def choose_unique_run_prefix(output_dir: Path, prefix: str) -> str:
    if not output_dir.exists():
        return prefix

    pattern = f"{prefix}*"
    if not any(output_dir.glob(pattern)):
        return prefix

    index = 2
    while True:
        candidate = f"{prefix}_run{index}"
        if not any(output_dir.glob(f"{candidate}*")):
            return candidate
        index += 1


def format_interval(interval: GenomicInterval) -> str:
    return f"{interval.chrom}:{interval.start:,}-{interval.end:,}"


def list_xlsx_candidates() -> list[Path]:
    roots = [Path("."), Path("Files"), Path("Outputs")]
    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*.xlsx"):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
    return sorted(found)


def parse_interval_from_any_text(raw: str) -> GenomicInterval | None:
    match = INTERVAL_SEARCH_RE.search(str(raw).strip())
    if not match:
        return None
    chrom = match.group("chrom")
    start = int(match.group("start").replace(",", ""))
    end = int(match.group("end").replace(",", ""))
    return GenomicInterval(chrom=chrom, start=min(start, end), end=max(start, end))


def normalize_chrom_token(raw: str) -> str | None:
    token = str(raw).strip()
    if not token:
        return None
    if not CHROM_TOKEN_RE.match(token):
        return None
    if not token.lower().startswith("chr"):
        token = f"chr{token}"
    return token


def extract_interval_tasks_from_headerless_sheet(frame: pd.DataFrame, sheet_name: str) -> list[IntervalTask]:
    tasks: list[IntervalTask] = []
    for idx, row in frame.iterrows():
        values = [value for value in row.values if not pd.isna(value)]
        if len(values) < 3:
            continue

        chrom_index = None
        chrom = None
        for j, value in enumerate(values):
            c = normalize_chrom_token(str(value))
            if c is not None:
                chrom_index = j
                chrom = c
                break
        if chrom_index is None or chrom is None:
            continue

        numeric_values: list[int] = []
        for value in values[chrom_index + 1 :]:
            try:
                numeric_values.append(int(float(value)))
            except (TypeError, ValueError):
                continue
        if len(numeric_values) < 2:
            continue

        start = numeric_values[0]
        end = numeric_values[1]
        if start < 1 or end < 1:
            continue

        label = None
        for value in values[:chrom_index]:
            text = str(value).strip()
            if text:
                label = text
                break
        if not label:
            label = f"{sheet_name}_row_{idx + 1}"

        tasks.append(
            IntervalTask(
                label=label,
                interval=GenomicInterval(chrom=chrom, start=min(start, end), end=max(start, end)),
            )
        )
    return tasks


def extract_interval_tasks_from_sheet(frame: pd.DataFrame, sheet_name: str) -> list[IntervalTask]:
    if frame.empty:
        return []

    tasks: list[IntervalTask] = []
    columns = {str(col).strip().lower(): col for col in frame.columns}

    interval_col_candidates = [
        "genomicintervals",
        "genomic interval",
        "interval",
        "coordinates",
        "coordinate",
        "region",
        "locus",
    ]
    label_col_candidates = [
        "acronym",
        "label",
        "name",
        "id",
        "enhancer",
        "gene",
        "overlap",
    ]

    interval_col = None
    for key in interval_col_candidates:
        if key in columns:
            interval_col = columns[key]
            break

    label_col = None
    for key in label_col_candidates:
        if key in columns:
            label_col = columns[key]
            break

    if interval_col is not None:
        for idx, row in frame.iterrows():
            value = row.get(interval_col)
            if pd.isna(value):
                continue
            interval = parse_interval_from_any_text(str(value))
            if interval is None:
                continue
            if label_col is not None and not pd.isna(row.get(label_col)):
                label = str(row.get(label_col)).strip()
            else:
                label = f"{sheet_name}_row_{idx + 2}"
            tasks.append(IntervalTask(label=label or f"{sheet_name}_row_{idx + 2}", interval=interval))
        if tasks:
            return tasks

    # Fallback 1: chrom/start/end style columns.
    chrom_col = None
    start_col = None
    end_col = None
    for key in ["chrom", "chr", "chromosome", "seq_region_name"]:
        if key in columns:
            chrom_col = columns[key]
            break
    for key in ["start", "chromstart", "interval_start"]:
        if key in columns:
            start_col = columns[key]
            break
    for key in ["end", "chromend", "interval_end"]:
        if key in columns:
            end_col = columns[key]
            break

    if chrom_col is not None and start_col is not None and end_col is not None:
        for idx, row in frame.iterrows():
            c = row.get(chrom_col)
            s = row.get(start_col)
            e = row.get(end_col)
            if pd.isna(c) or pd.isna(s) or pd.isna(e):
                continue
            chrom = str(c).strip()
            if not chrom.lower().startswith("chr"):
                chrom = f"chr{chrom}"
            try:
                start = int(float(s))
                end = int(float(e))
            except ValueError:
                continue
            interval = GenomicInterval(chrom=chrom, start=min(start, end), end=max(start, end))
            if label_col is not None and not pd.isna(row.get(label_col)):
                label = str(row.get(label_col)).strip()
            else:
                label = f"{sheet_name}_row_{idx + 2}"
            tasks.append(IntervalTask(label=label or f"{sheet_name}_row_{idx + 2}", interval=interval))
        if tasks:
            return tasks

    # Fallback 2: scan all text-like cells for interval strings.
    for idx, row in frame.iterrows():
        found_interval = None
        for value in row.values:
            if pd.isna(value):
                continue
            interval = parse_interval_from_any_text(str(value))
            if interval is not None:
                found_interval = interval
                break
        if found_interval is None:
            continue
        label = f"{sheet_name}_row_{idx + 2}"
        tasks.append(IntervalTask(label=label, interval=found_interval))

    return tasks


def interval_tasks_from_xlsx(path: Path) -> list[IntervalTask]:
    workbook = pd.read_excel(path, sheet_name=None)
    all_tasks: list[IntervalTask] = []
    for sheet_name, frame in workbook.items():
        all_tasks.extend(extract_interval_tasks_from_sheet(frame, sheet_name=sheet_name))

    # Fallback for headerless row-based workbooks where first data row is used as header.
    if not all_tasks:
        workbook_no_header = pd.read_excel(path, sheet_name=None, header=None)
        for sheet_name, frame in workbook_no_header.items():
            all_tasks.extend(extract_interval_tasks_from_headerless_sheet(frame, sheet_name=sheet_name))

    # Deduplicate by exact genomic coordinates, keep first label.
    deduped: list[IntervalTask] = []
    seen: set[tuple[str, int, int]] = set()
    for task in all_tasks:
        key = (task.interval.chrom, task.interval.start, task.interval.end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(task)

    if not deduped:
        raise RuntimeError("No valid intervals were detected in the XLSX file.")
    return deduped


def interval_tasks_from_user() -> list[IntervalTask]:
    print("\nHow would you like to provide intervals?")
    print("  1. Enter intervals manually")
    print("  2. Load intervals from an XLSX file")
    source_choice = prompt_int_range("Select option (1-2): ", minimum=1, maximum=2)

    if source_choice == 2:
        candidates = list_xlsx_candidates()
        if candidates:
            print("\nDetected XLSX files:")
            for p in candidates:
                print(f"  - {p}")
        path_text = prompt_nonempty("Path to XLSX file: ")
        xlsx_path = Path(path_text).expanduser()
        if not xlsx_path.exists() or not xlsx_path.is_file():
            raise FileNotFoundError(f"XLSX file not found: {xlsx_path}")
        tasks = interval_tasks_from_xlsx(xlsx_path)
        print(f"Loaded {len(tasks)} intervals from {xlsx_path}.")
        return tasks

    print("\nEnter one or more genomic intervals (e.g., chr1:100000-100500).")
    print("Press Enter on an empty line when finished.\n")

    tasks: list[IntervalTask] = []
    idx = 1
    while True:
        raw = input(f"Interval {idx}: ").strip()
        if not raw:
            break
        try:
            interval = parse_interval(raw)
        except ValueError as exc:
            print(exc)
            continue
        label_raw = input(f"Label for interval {idx} (optional): ").strip()
        label = label_raw if label_raw else f"interval_{idx}"
        tasks.append(IntervalTask(label=label, interval=interval))
        idx += 1

    if not tasks:
        raise RuntimeError("No intervals provided.")
    return tasks


def plot_interval_style(
    csv_path: Path,
    out_path: Path,
    xmin: float,
    xmax: float,
    title: str,
    subtitle: str,
    readout_rel_start: float,
    readout_rel_end: float,
) -> None:
    df = pd.read_csv(csv_path)
    required = {"position_rel_tss", "log2_fold_change", "seq_alt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    df["position_rel_tss"] = pd.to_numeric(df["position_rel_tss"], errors="coerce")
    df["log2_fold_change"] = pd.to_numeric(df["log2_fold_change"], errors="coerce")
    df = df.dropna(subset=["position_rel_tss", "log2_fold_change"]).copy()
    df = df[(df["position_rel_tss"] >= xmin) & (df["position_rel_tss"] <= xmax)].copy()
    if df.empty:
        raise RuntimeError("No rows remain after filtering to requested x-range")

    df["seq_alt"] = df["seq_alt"].astype(str).str.upper()
    base_rank = {"A": 0, "C": 1, "G": 2, "T": 3}
    df["_alt_rank"] = df["seq_alt"].map(base_rank).fillna(9)
    df = df.sort_values(["position_rel_tss", "_alt_rank"]).reset_index(drop=True)
    df["_x"] = df["position_rel_tss"] + df["seq_alt"].map(X_OFFSETS).fillna(0.0)

    colors = [MUTATION_COLORS.get(base, "#777777") for base in df["seq_alt"]]

    if "p_value" in df.columns:
        sig_mask = pd.to_numeric(df["p_value"], errors="coerce") < 0.05
        sig_label = "dot: p < 0.05"
    elif "is_significant_p05" in df.columns:
        sig_mask = pd.Series(df["is_significant_p05"]).fillna(False).astype(bool)
        sig_label = "dot: significant"
    else:
        sig_mask = pd.Series(False, index=df.index)
        sig_label = "dot: significant"

    fig, ax = plt.subplots(1, 1, figsize=(16, 6), constrained_layout=True)

    ax.vlines(
        df["_x"].to_numpy(dtype=float),
        0.0,
        df["log2_fold_change"].to_numpy(dtype=float),
        colors=colors,
        linewidth=0.85,
        alpha=0.7,
    )

    sig_rows = df.loc[sig_mask]
    if not sig_rows.empty:
        sig_colors = [MUTATION_COLORS.get(base, "#777777") for base in sig_rows["seq_alt"]]
        ax.scatter(sig_rows["_x"], sig_rows["log2_fold_change"], s=16, c=sig_colors, zorder=3, alpha=0.95)

    ax.axhline(0.0, color="#666666", lw=1.0)
    ax.axvline(0.0, color="#999999", lw=1.0, ls="--")
    ro_left = min(float(readout_rel_start), float(readout_rel_end))
    ro_right = max(float(readout_rel_start), float(readout_rel_end))
    ax.axvspan(ro_left, ro_right, color="#1f77b4", alpha=0.12, zorder=0)
    ax.grid(axis="y", alpha=0.25)

    ax.set_xlim(xmin, xmax)
    ax.set_ylabel("Log2 fold change")
    ax.set_xlabel("Position relative to interval midpoint (bp)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))

    full_title = title
    if subtitle:
        full_title = f"{title}\n{subtitle}"
    ax.set_title(full_title)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=7, label=f"WT->{base}")
        for base, color in (("A", MUTATION_COLORS["A"]), ("T", MUTATION_COLORS["T"]), ("C", MUTATION_COLORS["C"]), ("G", MUTATION_COLORS["G"]))
    ]
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#111111", markersize=6, label=sig_label)
    )
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, ncol=5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive interval saturation mutagenesis runner.")
    parser.add_argument("--api-key", default=None, help="Optional AlphaGenome API key. Falls back to ALPHAGENOME_API_KEY.")
    parser.add_argument("--output-root", default="Outputs", help="Root output directory (default: Outputs)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\nInteractive interval-centered saturation mutagenesis\n")

    acronym = prompt_nonempty("Project acronym for output folder (e.g., MYSET): ").upper()
    tasks = interval_tasks_from_user()

    normalize_magnitude = prompt_yes_no(
        "Normalize scores by readout length for cross-run comparability?",
        default_yes=True,
    )
    position_aggregation = "mean" if normalize_magnitude else "sum"

    dataset_key, dataset_label, ontology_terms, post_filter_keywords = prompt_dataset_choice()
    output_label, output_types, output_filter = prompt_output_choice()
    output_filter = prompt_output_filter()

    api_key = args.api_key or os.getenv("ALPHAGENOME_API_KEY")
    if not api_key:
        api_key = prompt_nonempty("AlphaGenome API key: ")

    project_dir = Path(args.output_root) / acronym
    project_dir.mkdir(parents=True, exist_ok=True)

    print("\nRun summary:")
    print(f"  Project: {acronym}")
    print(f"  Intervals: {len(tasks)}")
    print(f"  Position aggregation: {position_aggregation}")
    print(f"  Dataset filter: {dataset_label}")
    print(f"  Output preset: {output_label}")
    print(f"  Output types: {', '.join(output_types)}")
    print(f"  Output filter: {output_filter if output_filter else 'none'}")
    print("  Track grouping: biosample")
    print(f"  Ontology terms: {', '.join(ontology_terms) if ontology_terms else 'none'}")
    if post_filter_keywords:
        print(f"  Strict biosample keywords: {', '.join(post_filter_keywords)}")
    for task in tasks:
        print(f"  - {task.label}: {format_interval(task.interval)}")

    if not prompt_yes_no("Proceed with this run?", default_yes=True):
        print("Run cancelled by user before scoring.")
        return

    run_lines = [
        "Run completion summary:",
        f"  Completed at: {datetime.now().isoformat(timespec='seconds')}",
        f"  Project: {acronym}",
        f"  Position aggregation: {position_aggregation}",
        f"  Dataset filter: {dataset_label}",
        f"  Output preset: {output_label}",
        f"  Output types: {', '.join(output_types)}",
        "  Track grouping: biosample",
        f"  Ontology terms: {', '.join(ontology_terms) if ontology_terms else 'none'}",
        f"  Strict biosample keywords: {', '.join(post_filter_keywords) if post_filter_keywords else 'none'}",
        f"  Interval count: {len(tasks)}",
    ]

    for task in tasks:
        interval = task.interval
        center = (interval.start + interval.end) // 2
        upstream = center - interval.start
        downstream = interval.end - center

        rel_left = float(interval.start - center)
        rel_right = float(interval.end - center)

        safe_label = slugify(task.label)
        out_dir = project_dir / safe_label
        plots_dir = out_dir / "Plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        prefix_base = f"{acronym.lower()}_{safe_label}_{interval.chrom}_{interval.start}_{interval.end}"
        prefix = choose_unique_run_prefix(out_dir, prefix_base)

        context_fasta = out_dir / f"{prefix}_context_150kb_hg38.fa"
        window_fasta = out_dir / f"{prefix}_window.fa"
        table_csv = out_dir / f"{prefix}_variant_effects.csv"
        raw_plot = plots_dir / f"{prefix}_raw.png"
        style_plot = plots_dir / f"{prefix}_style_p_lt_0.05_only.png"
        metadata_summary_csv = out_dir / f"{prefix}_metadata_summary.csv"

        cmd = [
            sys.executable,
            "tss_saturation_pipeline.py",
            "--tss",
            f"{interval.chrom}:{center}",
            "--strand",
            "+",
            "--upstream",
            str(upstream),
            "--downstream",
            str(downstream),
            "--flank",
            "150000",
            "--api-key",
            api_key,
            "--context-fasta-out",
            str(context_fasta),
            "--window-fasta-out",
            str(window_fasta),
            "--outfile",
            str(table_csv),
            "--plot-out",
            str(raw_plot),
            "--readout-start-genomic",
            str(interval.start),
            "--readout-end-genomic",
            str(interval.end),
            "--position-aggregation",
            position_aggregation,
            "--track-grouping",
            "biosample",
            "--output-types",
            *output_types,
            "--batch-size",
            str(args.batch_size),
            "--max-workers",
            str(args.max_workers),
            "--progress-log-every",
            "1",
            "--metadata-summary-out",
            str(metadata_summary_csv),
        ]
        if ontology_terms:
            if dataset_key == "custom":
                # For arbitrary user-entered CURIEs, avoid request-time ontology
                # filtering (which may reject unknown IDs) and filter locally.
                cmd.extend(["--post-filter-ontology-curies", *ontology_terms])
            else:
                cmd.extend(["--ontology-terms", *ontology_terms])
        if post_filter_keywords:
            cmd.extend(["--post-filter-biosample-keywords", *post_filter_keywords])
        if output_filter:
            cmd.extend(["--output-filter", output_filter])

        print(f"\nRunning {task.label} ({format_interval(interval)})...")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        subtitle = (
            f"interval: {interval.chrom}:{interval.start:,}-{interval.end:,} | "
            f"readout: full interval"
        )
        plot_interval_style(
            csv_path=table_csv,
            out_path=style_plot,
            xmin=float(rel_left),
            xmax=float(rel_right),
            title=f"{acronym} {task.label} SNV effects",
            subtitle=subtitle,
            readout_rel_start=rel_left,
            readout_rel_end=rel_right,
        )

        run_lines.extend(
            [
                f"  Interval label: {task.label}",
                f"    Genomic interval: {format_interval(interval)}",
                f"    Midpoint: {interval.chrom}:{center:,}",
                f"    Relative window: {int(rel_left)}/{int(rel_right)}",
                f"    Metadata summary: {metadata_summary_csv}",
                f"    Data table: {table_csv}",
                f"    Raw plot: {raw_plot}",
                f"    Styled plot: {style_plot}",
            ]
        )

    summary_text = "\n".join(run_lines)
    summary_file = project_dir / f"{acronym.lower()}_run_summary.txt"
    summary_file.write_text(summary_text + "\n", encoding="utf-8")

    print("\nDone.")
    print(f"Project folder: {project_dir}")
    print(f"Summary file: {summary_file}")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
