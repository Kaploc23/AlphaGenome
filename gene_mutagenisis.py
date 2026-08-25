#!/usr/bin/env python3
"""Interactive runner for TSS-centered saturation mutagenesis and DSP-style plotting.

This script prompts for:
- gene symbol
- UTR interval (optionally with strand)
- strand if not included in the interval
- mutagenesis window upstream/downstream sizes
- optional UTR highlight
- optional TF interval highlights

Then it runs tss_saturation_pipeline.py with:
- 150 kb context
- promoter-wide readout window
- 1% progress logging

Finally it writes:
- scored CSV (with p-values)
- raw pipeline plot
- DSP-style annotated plot
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import urllib.error
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from Bio.Seq import Seq
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
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

INTERVAL_RE = re.compile(r"^(?P<chrom>[^:]+):(?P<start>[0-9,]+)-(?P<end>[0-9,]+)(?:\((?P<strand>[+-])\))?$")
JASPAR_THRESHOLD_FRACTION = 0.80
JASPAR_MAX_INTERVALS_PER_TF = 10

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
class Region:
    name: str
    interval: GenomicInterval


@dataclass
class TFSelection:
    regions: list[Region]
    ordered_names: list[str]


@dataclass
class GeneAnnotation:
    gene_symbol: str
    transcript_id: str
    chrom: str
    strand: str
    tss_1based: int
    utr_interval: GenomicInterval
    source: str


def parse_interval(text: str) -> tuple[GenomicInterval, str | None]:
    raw = text.strip()
    match = INTERVAL_RE.match(raw)
    if not match:
        raise ValueError(f"Could not parse interval: {text}")
    chrom = match.group("chrom")
    start = int(match.group("start").replace(",", ""))
    end = int(match.group("end").replace(",", ""))
    strand = match.group("strand")
    if start < 1 or end < 1:
        raise ValueError("Interval positions must be >= 1")
    return GenomicInterval(chrom=chrom, start=min(start, end), end=max(start, end)), strand


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


def prompt_strand(prompt: str) -> str:
    while True:
        raw = input(prompt).strip()
        if raw in {"+", "-"}:
            return raw
        print("Please enter + or -.")


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
        # For some presets we also want to propagate a high-level filter key
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


# (output filter selection now integrated into prompt_output_choice)


def tss_from_utr(utr: GenomicInterval, strand: str) -> int:
    if strand == "+":
        return utr.start
    return utr.end


def rel_coord(pos_1based: int, tss_1based: int, strand: str) -> float:
    if strand == "+":
        return float(pos_1based - tss_1based)
    return float(tss_1based - pos_1based)


def interval_to_rel(interval: GenomicInterval, tss_1based: int, strand: str) -> tuple[float, float]:
    left = rel_coord(interval.start, tss_1based, strand)
    right = rel_coord(interval.end, tss_1based, strand)
    return min(left, right), max(left, right)


def mutagenesis_interval(chrom: str, tss_1based: int, strand: str, upstream: int, downstream: int) -> GenomicInterval:
    if strand == "+":
        start = tss_1based - upstream
        end = tss_1based + downstream
    else:
        start = tss_1based - downstream
        end = tss_1based + upstream
    return GenomicInterval(chrom=chrom, start=min(start, end), end=max(start, end))


def choose_unique_run_prefix(output_dir: Path, prefix: str) -> str:
    """Return a non-conflicting prefix by appending _runN when needed."""
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


def summarize_tf_intervals(selection: TFSelection) -> list[str]:
    if not selection.ordered_names:
        return ["  TF highlights: none"]

    grouped: dict[str, list[GenomicInterval]] = {}
    for region in selection.regions:
        grouped.setdefault(region.name, []).append(region.interval)

    lines = [f"  TF highlights: {len(selection.ordered_names)} TFs"]
    for name in selection.ordered_names:
        intervals = grouped.get(name, [])
        intervals = sorted(intervals, key=lambda iv: (iv.start, iv.end))
        interval_text = "; ".join(format_interval(iv) for iv in intervals)
        if not interval_text:
            interval_text = "(no intervals)"
        lines.append(f"    - {name}: {interval_text}")
    return lines


def prompt_tf_include_names() -> list[str] | None:
    if not prompt_yes_no("Limit TF highlights to specific names?", default_yes=False):
        return None

    raw = prompt_nonempty("Enter TF names (comma-separated, e.g., GATA6,GATA4): ")
    names = [normalize_tf_name(token) for token in raw.split(",") if token.strip()]
    names = list(dict.fromkeys(name for name in names if name))
    if not names:
        print("No valid TF names provided; using all selected TFs.")
        return None
    print("Will include only these TFs:", ", ".join(names))
    return names


def filter_tf_selection_by_include_names(selection: TFSelection, include_names: list[str] | None) -> TFSelection:
    if not include_names:
        return selection
    if not selection.ordered_names:
        return selection

    include_set = {normalize_tf_name(name) for name in include_names if str(name).strip()}
    if not include_set:
        return selection

    kept_names = [
        name
        for name in selection.ordered_names
        if tf_name_aliases(name) & include_set
    ]
    kept_name_set = set(kept_names)
    kept_regions = [region for region in selection.regions if region.name in kept_name_set]

    matched_inputs = sorted({alias for name in kept_names for alias in tf_name_aliases(name) if alias in include_set})
    missing_inputs = [name for name in include_names if name not in matched_inputs]

    print(f"Filtered TF highlights to requested names: {len(kept_names)} TFs, {len(kept_regions)} intervals")
    if missing_inputs:
        print("Requested TFs not found in selected intervals:", ", ".join(missing_inputs))

    return TFSelection(regions=kept_regions, ordered_names=kept_names)


def resolve_heart_tf_path(files_root: Path) -> Path:
    candidates = [
        files_root / "Misc" / "HeartTSList.xlsx",
        files_root / "HeartTSList.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_tf_lines(lines: list[str], expected_chrom: str) -> list[Region]:
    regions: list[Region] = []
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        if "," not in raw:
            print(f"Skipping malformed TF line (missing comma): {raw}")
            continue
        tf_name, interval_block = raw.split(",", 1)
        tf_name = tf_name.strip()
        if not tf_name:
            print(f"Skipping malformed TF line (empty TF name): {raw}")
            continue
        for part in interval_block.split(";"):
            chunk = part.strip()
            if not chunk:
                continue
            try:
                interval, _ = parse_interval(chunk)
            except ValueError:
                print(f"Skipping malformed TF interval: {chunk}")
                continue
            if interval.chrom != expected_chrom:
                print(f"Skipping interval on different chromosome: {chunk}")
                continue
            regions.append(Region(name=tf_name, interval=interval))
    return regions


def pick_best_transcript(rows: pd.DataFrame) -> pd.Series:
    frame = rows.copy()
    frame["has_utr"] = frame["5_utr_start"].notna() & frame["5_utr_end"].notna()
    frame["utr_len"] = (pd.to_numeric(frame["5_utr_end"], errors="coerce") - pd.to_numeric(frame["5_utr_start"], errors="coerce")).abs() + 1
    if "transcript_is_canonical" in frame.columns:
        frame["is_canonical"] = pd.to_numeric(frame["transcript_is_canonical"], errors="coerce").fillna(0)
    else:
        frame["is_canonical"] = 0

    if "transcript_biotype" in frame.columns:
        frame["is_protein_coding"] = frame["transcript_biotype"].astype(str).str.lower().eq("protein_coding")
    else:
        frame["is_protein_coding"] = False

    frame = frame.sort_values(
        by=["has_utr", "is_canonical", "is_protein_coding", "utr_len"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return frame.iloc[0]


def lookup_ensembl_gene_id(gene_symbol: str) -> str:
    url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}?content-type=application/json"
    headers = {"Accept": "application/json", "User-Agent": "AlphaGenome/1.0 (contact: none)"}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if "id" not in data:
        raise RuntimeError(f"Could not resolve Ensembl gene ID for symbol: {gene_symbol}")
    return str(data["id"])


def normalize_chromosome(chromosome_name: str) -> str:
    value = str(chromosome_name).strip()
    if not value:
        raise RuntimeError("Missing chromosome name in annotation response")
    if value.lower().startswith("chr"):
        return value
    if value.upper() == "MT":
        return "chrM"
    return f"chr{value}"


def rest_get_json(path: str) -> tuple[dict | list, str]:
    hosts = [
        "https://rest.ensembl.org",
        "https://useast.ensembl.org",
    ]
    last_error: Exception | None = None

    # Include a User-Agent to avoid naive blocking by some Ensembl mirrors.
    headers = {"Accept": "application/json", "User-Agent": "AlphaGenome/1.0 (contact: none)"}
    for host in hosts:
        url = f"{host}{path}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
            data = json.loads(payload)
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(str(data.get("error")))
            return data, host
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Ensembl REST request failed for path {path}. Last error: {last_error}")


def fetch_hg38_sequence_for_interval(interval: GenomicInterval) -> str:
    # UCSC endpoint uses 0-based half-open coordinates.
    start0 = interval.start - 1
    end0 = interval.end
    url = (
        "https://api.genome.ucsc.edu/getData/sequence"
        f"?genome=hg38;chrom={interval.chrom};start={start0};end={end0}"
    )
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    seq = str(data.get("dna", "")).upper()
    expected = interval.end - interval.start + 1
    if len(seq) != expected:
        raise RuntimeError(
            f"UCSC sequence fetch length mismatch for {interval.chrom}:{interval.start}-{interval.end}: "
            f"expected {expected}, got {len(seq)}"
        )
    return seq


def pick_best_rest_transcript(transcripts: list[dict], canonical_transcript_id: str | None) -> dict:
    if not transcripts:
        raise RuntimeError("No transcripts returned by Ensembl REST")

    rows: list[dict] = []
    canonical_base = (canonical_transcript_id or "").split(".", 1)[0]
    for tx in transcripts:
        tx_id = str(tx.get("id", ""))
        is_canonical = bool(tx.get("is_canonical")) or (tx_id == canonical_base)
        biotype = str(tx.get("biotype", "")).lower()
        is_protein_coding = biotype == "protein_coding"
        tr = tx.get("Translation") or {}
        has_translation = tr.get("start") is not None and tr.get("end") is not None
        length = int(tx.get("length") or 0)
        rows.append(
            {
                "tx": tx,
                "is_canonical": is_canonical,
                "is_protein_coding": is_protein_coding,
                "has_translation": has_translation,
                "length": length,
            }
        )

    rows.sort(
        key=lambda item: (
            item["is_canonical"],
            item["is_protein_coding"],
            item["has_translation"],
            item["length"],
        ),
        reverse=True,
    )
    return rows[0]["tx"]


def parse_required_int(value: object, field_name: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing or invalid integer field '{field_name}' in Ensembl REST response") from exc
    return parsed


def derive_tss_and_utr_from_rest_transcript(transcript: dict) -> tuple[str, int, GenomicInterval]:
    strand_raw = parse_required_int(transcript.get("strand"), "strand")
    strand = "+" if strand_raw == 1 else "-"

    tx_start = parse_required_int(transcript.get("start"), "start")
    tx_end = parse_required_int(transcript.get("end"), "end")
    translation = transcript.get("Translation") or {}
    cds_start = parse_required_int(translation.get("start"), "Translation.start")
    cds_end = parse_required_int(translation.get("end"), "Translation.end")

    if strand == "+":
        tss_1based = tx_start
        utr_start = tx_start
        utr_end = cds_start - 1
    else:
        tss_1based = tx_end
        utr_start = cds_end + 1
        utr_end = tx_end

    if utr_end < utr_start:
        raise RuntimeError(f"Transcript {transcript.get('id')} has no detectable 5' UTR")

    chrom = normalize_chromosome(str(transcript.get("seq_region_name", "")))
    utr_interval = GenomicInterval(chrom=chrom, start=min(utr_start, utr_end), end=max(utr_start, utr_end))
    return strand, tss_1based, utr_interval


def lookup_gene_annotation_with_ensembl_rest(gene_symbol: str) -> GeneAnnotation:
    symbol = gene_symbol.strip().upper()
    encoded = urllib.parse.quote(symbol)
    payload, host = rest_get_json(f"/lookup/symbol/homo_sapiens/{encoded}?expand=1")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected Ensembl REST response type for symbol {symbol}")

    transcripts_raw = payload.get("Transcript")
    if not isinstance(transcripts_raw, list) or not transcripts_raw:
        raise RuntimeError(f"No transcripts found for gene {symbol}")

    allowed_chr = {str(x) for x in list(range(1, 23)) + ["X", "Y", "MT", "M"]}
    transcripts = [
        tx
        for tx in transcripts_raw
        if str(tx.get("seq_region_name", "")).upper() in allowed_chr
    ]
    if not transcripts:
        raise RuntimeError(f"No transcripts on main chromosomes found for gene {symbol}")

    best = pick_best_rest_transcript(transcripts, canonical_transcript_id=str(payload.get("canonical_transcript", "")))
    strand, tss_1based, utr_interval = derive_tss_and_utr_from_rest_transcript(best)

    return GeneAnnotation(
        gene_symbol=symbol,
        transcript_id=str(best.get("id", "")),
        chrom=utr_interval.chrom,
        strand=strand,
        tss_1based=tss_1based,
        utr_interval=utr_interval,
        source=f"ensembl-rest:{host}",
    )


def lookup_gene_annotation_with_biomart(gene_symbol: str) -> GeneAnnotation:
    try:
        from pybiomart import Server
    except Exception as exc:
        raise RuntimeError("pybiomart is not installed. Run: pip install pybiomart") from exc

    attrs = [
        "hgnc_symbol",
        "ensembl_transcript_id",
        "chromosome_name",
        "strand",
        "transcription_start_site",
        "5_utr_start",
        "5_utr_end",
        "transcript_is_canonical",
        "transcript_biotype",
    ]

    hosts = [
        "http://www.ensembl.org",
        "http://useast.ensembl.org",
        "http://asia.ensembl.org",
    ]
    symbol = gene_symbol.strip().upper()
    last_error: Exception | None = None

    for host in hosts:
        try:
            server = Server(host=host)
            dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]

            # Try direct xref symbol lookup first.
            try:
                table = dataset.query(
                    attributes=attrs,
                    filters={"id_list_xrefs_filters": [symbol]},
                    use_attr_names=True,
                )
            except TypeError:
                table = dataset.query(attributes=attrs, filters={"id_list_xrefs_filters": [symbol]})

            # Biomart outage responses can come back as HTML tables.
            if table.shape[1] == 1 and str(table.columns[0]).lower().startswith("<html"):
                raise RuntimeError("Biomart returned HTML service-unavailable response")

            if "hgnc_symbol" in table.columns:
                table = table[table["hgnc_symbol"].astype(str).str.upper() == symbol].copy()

            # Fallback path: resolve Ensembl gene ID, then query by stable ID.
            if table.empty:
                gene_id = lookup_ensembl_gene_id(symbol)
                try:
                    table = dataset.query(
                        attributes=attrs,
                        filters={"gene_id": [gene_id]},
                        use_attr_names=True,
                    )
                except TypeError:
                    table = dataset.query(attributes=attrs, filters={"gene_id": [gene_id]})

            if table.shape[1] == 1 and str(table.columns[0]).lower().startswith("<html"):
                raise RuntimeError("Biomart returned HTML service-unavailable response")

            if table.empty:
                continue

            # Restrict to main chromosomes.
            allowed_chr = {str(x) for x in list(range(1, 23)) + ["X", "Y", "MT", "M"]}
            table = table[table["chromosome_name"].astype(str).isin(allowed_chr)].copy()
            if table.empty:
                continue

            best = pick_best_transcript(table)
            strand_val = int(pd.to_numeric(best["strand"], errors="coerce"))
            strand = "+" if strand_val == 1 else "-"

            tss_raw = int(pd.to_numeric(best["transcription_start_site"], errors="coerce"))
            utr_start = int(pd.to_numeric(best["5_utr_start"], errors="coerce"))
            utr_end = int(pd.to_numeric(best["5_utr_end"], errors="coerce"))

            interval = GenomicInterval(
                chrom=f"chr{best['chromosome_name']}",
                start=min(utr_start, utr_end),
                end=max(utr_start, utr_end),
            )

            return GeneAnnotation(
                gene_symbol=symbol,
                transcript_id=str(best["ensembl_transcript_id"]),
                chrom=interval.chrom,
                strand=strand,
                tss_1based=tss_raw,
                utr_interval=interval,
                source=f"pybiomart:{host}",
            )
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Could not determine TSS/UTR via pybiomart for gene {symbol}. Last error: {last_error}")


def normalize_tf_name(name: str) -> str:
    return str(name).strip().upper()


def tf_name_aliases(name: str) -> set[str]:
    norm = normalize_tf_name(name)
    aliases = {norm}
    if "::" in norm:
        aliases.update(part.strip() for part in norm.split("::") if part.strip())
    return aliases


def tf_in_heart(name: str, heart_tf_names: set[str]) -> bool:
    return any(alias in heart_tf_names for alias in tf_name_aliases(name))


def chromosome_for_ensembl_region(chrom: str) -> str:
    token = str(chrom).strip()
    if token.lower().startswith("chr"):
        token = token[3:]
    token = token.upper()
    if token == "M":
        return "MT"
    return token


def fetch_ensembl_tf_regions(interval: GenomicInterval) -> list[Region]:
    chrom = chromosome_for_ensembl_region(interval.chrom)
    path = f"/overlap/region/homo_sapiens/{chrom}:{interval.start}-{interval.end}?feature=motif"
    payload, _ = rest_get_json(path)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Ensembl REST motif payload format")

    regions: list[Region] = []
    seen: set[tuple[str, int, int]] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        tf_complex = str(item.get("transcription_factor_complex", "")).strip()
        if not tf_complex:
            continue
        start = parse_required_int(item.get("start"), "motif.start")
        end = parse_required_int(item.get("end"), "motif.end")
        for token in tf_complex.split(","):
            tf_name = normalize_tf_name(token)
            if not tf_name:
                continue
            key = (tf_name, min(start, end), max(start, end))
            if key in seen:
                continue
            seen.add(key)
            regions.append(
                Region(
                    name=tf_name,
                    interval=GenomicInterval(
                        chrom=interval.chrom,
                        start=min(start, end),
                        end=max(start, end),
                    ),
                )
            )
    return regions


def fetch_jaspar_tf_regions(interval: GenomicInterval, heart_tf_names: set[str] | None = None) -> list[Region]:
    try:
        from pyjaspar import jaspardb
    except Exception as exc:
        raise RuntimeError("pyjaspar is not installed. Run: pip install pyjaspar biopython") from exc

    sequence = fetch_hg38_sequence_for_interval(interval)
    seq_obj = Seq(sequence)

    db = jaspardb(release="JASPAR2024")
    motifs = db.fetch_motifs(collection="CORE", tax_group=["vertebrates"])

    if heart_tf_names:
        candidate_motifs = [m for m in motifs if normalize_tf_name(getattr(m, "name", "")) in heart_tf_names]
    else:
        candidate_motifs = list(motifs)
    if not candidate_motifs:
        return []

    hits_by_tf: dict[str, list[tuple[float, int, int]]] = {}
    for motif in candidate_motifs:
        tf_name = normalize_tf_name(getattr(motif, "name", ""))
        if not tf_name:
            continue

        pssm = motif.pwm.log_odds()
        threshold = pssm.min + JASPAR_THRESHOLD_FRACTION * (pssm.max - pssm.min)
        motif_len = len(motif)

        for pos, score in pssm.search(seq_obj, threshold=threshold, both=False):
            start = interval.start + int(pos)
            end = start + motif_len - 1
            hits_by_tf.setdefault(tf_name, []).append((float(score), start, end))

        rc_pssm = pssm.reverse_complement()
        for pos, score in rc_pssm.search(seq_obj, threshold=threshold, both=False):
            start = interval.start + int(pos)
            end = start + motif_len - 1
            hits_by_tf.setdefault(tf_name, []).append((float(score), start, end))

    regions: list[Region] = []
    for tf_name, hits in hits_by_tf.items():
        # Keep strongest matches per TF so the plot remains readable.
        dedup: dict[tuple[int, int], float] = {}
        for score, start, end in hits:
            key = (start, end)
            prev = dedup.get(key)
            if prev is None or score > prev:
                dedup[key] = score

        top_hits = sorted(((s, st, en) for (st, en), s in dedup.items()), reverse=True)
        for score, start, end in top_hits[:JASPAR_MAX_INTERVALS_PER_TF]:
            _ = score
            regions.append(
                Region(
                    name=tf_name,
                    interval=GenomicInterval(
                        chrom=interval.chrom,
                        start=min(start, end),
                        end=max(start, end),
                    ),
                )
            )

    return regions


def select_tf_regions_by_heart_order(all_regions: list[Region], heart_xlsx: Path) -> TFSelection:
    if not all_regions:
        return TFSelection(regions=[], ordered_names=[])

    heart_tf_order = load_heart_tf_order(heart_xlsx)
    if not heart_tf_order:
        print("No TF names found in HeartTSList.xlsx; skipping TF highlighting.")
        return TFSelection(regions=[], ordered_names=[])

    counts_by_name = Counter(region.name for region in all_regions)
    source_tf_order = list(dict.fromkeys(region.name for region in all_regions))

    heart_to_source_names: dict[str, list[str]] = {}
    for heart_name in heart_tf_order:
        matched: list[str] = []
        for source_name in source_tf_order:
            if heart_name in tf_name_aliases(source_name):
                matched.append(source_name)
        if matched:
            heart_to_source_names[heart_name] = matched

    overlap_names: list[str] = []
    used_source_names: set[str] = set()
    for heart_name in heart_tf_order:
        for source_name in heart_to_source_names.get(heart_name, []):
            if source_name in used_source_names:
                continue
            used_source_names.add(source_name)
            overlap_names.append(source_name)

    print(f"Unique TFs in source intervals: {len(counts_by_name)}")
    print(f"Total TF intervals: {len(all_regions)}")
    print(f"TFs overlapping HeartTSList: {len(overlap_names)}")

    if not overlap_names:
        print("No overlapping TFs found for HeartTSList ranking.")
        return TFSelection(regions=[], ordered_names=[])

    max_n = len(overlap_names)
    n_to_include = prompt_int_range(
        f"How many overlapping TFs to include on plot? (1-{max_n}): ",
        minimum=1,
        maximum=max_n,
    )

    selected_in_order = overlap_names[:n_to_include]
    selected_names = set(selected_in_order)
    selected_regions = [region for region in all_regions if region.name in selected_names]

    print("Selected TFs:")
    for name in selected_in_order:
        print(f"  - {name} ({counts_by_name[name]} intervals)")

    return TFSelection(regions=selected_regions, ordered_names=selected_in_order)


def select_tf_regions_by_frequency(all_regions: list[Region], source_label: str) -> TFSelection:
    if not all_regions:
        return TFSelection(regions=[], ordered_names=[])

    counts_by_name = Counter(region.name for region in all_regions)
    ordered_names = sorted(counts_by_name.keys(), key=lambda n: (-counts_by_name[n], n))

    print(f"Unique TFs in {source_label}: {len(ordered_names)}")
    print(f"Total TF intervals in {source_label}: {len(all_regions)}")

    max_n = len(ordered_names)
    n_to_include = prompt_int_range(
        f"How many TFs to include on plot? (1-{max_n}): ",
        minimum=1,
        maximum=max_n,
    )

    selected_in_order = ordered_names[:n_to_include]
    selected_names = set(selected_in_order)
    selected_regions = [region for region in all_regions if region.name in selected_names]

    print("Selected TFs:")
    for name in selected_in_order:
        print(f"  - {name} ({counts_by_name[name]} intervals)")

    return TFSelection(regions=selected_regions, ordered_names=selected_in_order)


def load_heart_tf_names(heart_xlsx: Path) -> set[str]:
    if not heart_xlsx.exists():
        raise FileNotFoundError(f"Heart TF reference not found: {heart_xlsx}")

    workbook = pd.read_excel(heart_xlsx, sheet_name=None)
    preferred_columns = ["Sample", "TF", "TFName", "Gene", "GeneSymbol", "Symbol"]

    tf_names: set[str] = set()
    for _, frame in workbook.items():
        columns_to_scan: list[str] = [col for col in preferred_columns if col in frame.columns]
        if not columns_to_scan:
            object_columns = [col for col in frame.columns if frame[col].dtype == object]
            if object_columns:
                columns_to_scan = [object_columns[0]]

        for col in columns_to_scan:
            values = frame[col].dropna().astype(str)
            for value in values:
                norm = normalize_tf_name(value)
                if norm:
                    tf_names.add(norm)

    return tf_names


def load_heart_tf_order(heart_xlsx: Path) -> list[str]:
    if not heart_xlsx.exists():
        raise FileNotFoundError(f"Heart TF reference not found: {heart_xlsx}")

    workbook = pd.read_excel(heart_xlsx, sheet_name=None)
    preferred_columns = ["Sample", "TF", "TFName", "Gene", "GeneSymbol", "Symbol"]

    ordered: list[str] = []
    seen: set[str] = set()
    for _, frame in workbook.items():
        columns_to_scan: list[str] = [col for col in preferred_columns if col in frame.columns]
        if not columns_to_scan:
            object_columns = [col for col in frame.columns if frame[col].dtype == object]
            if object_columns:
                columns_to_scan = [object_columns[0]]

        for col in columns_to_scan:
            values = frame[col].dropna().astype(str)
            for value in values:
                norm = normalize_tf_name(value)
                if norm and norm not in seen:
                    seen.add(norm)
                    ordered.append(norm)

    return ordered


def read_tf_csv(tf_file: Path) -> pd.DataFrame:
    skiprows = 0
    with tf_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                skiprows += 1
                continue
            if line.lower().startswith("track "):
                skiprows += 1
                continue
            break
    return pd.read_csv(tf_file, skiprows=skiprows)


def load_tf_regions_from_file(tf_file: Path, expected_chrom: str) -> list[Region]:
    table = read_tf_csv(tf_file)
    cols = {str(c).strip() for c in table.columns}

    if {"Overlap", "GenomicIntervals"}.issubset(cols):
        lines = [f"{row['Overlap']},{row['GenomicIntervals']}" for _, row in table.iterrows()]
        return parse_tf_lines(lines, expected_chrom=expected_chrom)

    required_track_cols = {"chrom", "chromStart", "chromEnd", "TFName"}
    if required_track_cols.issubset(cols):
        regions: list[Region] = []
        for _, row in table.iterrows():
            chrom = str(row["chrom"]).strip().strip('"')
            if chrom != expected_chrom:
                continue
            tf_name = str(row["TFName"]).strip()
            if not tf_name or tf_name.lower() == "nan":
                continue
            try:
                start0 = int(row["chromStart"])
                end0 = int(row["chromEnd"])
            except (TypeError, ValueError):
                continue
            if end0 <= start0:
                continue
            interval = GenomicInterval(chrom=chrom, start=start0 + 1, end=end0)
            regions.append(Region(name=tf_name, interval=interval))
        return regions

    raise ValueError(
        "TF CSV format not recognized. Expected either columns Overlap+GenomicIntervals "
        "or chrom+chromStart+chromEnd+TFName."
    )


def plot_dsp_style(
    csv_path: Path,
    out_path: Path,
    tss_1based: int,
    strand: str,
    xmin: float,
    xmax: float,
    title: str,
    subtitle: str,
    readout_rel_start: float,
    readout_rel_end: float,
    highlight_utr: bool,
    utr_region: Region | None,
    tf_regions: list[Region],
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

    region_entries: list[tuple[str, float, float]] = []
    if highlight_utr and utr_region is not None:
        left, right = interval_to_rel(utr_region.interval, tss_1based, strand)
        if not (right < xmin or left > xmax):
            region_entries.append((utr_region.name, float(left), float(right)))

    for region in tf_regions:
        left, right = interval_to_rel(region.interval, tss_1based, strand)
        if right < xmin or left > xmax:
            continue
        region_entries.append((region.name, float(left), float(right)))

    if region_entries:
        fig, (ax, ax_regions) = plt.subplots(
            2,
            1,
            figsize=(16, 7.8),
            constrained_layout=True,
            sharex=True,
            gridspec_kw={"height_ratios": [8.5, 1.8]},
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(16, 6), constrained_layout=True)
        ax_regions = None

    ax.vlines(df["_x"].to_numpy(dtype=float), 0.0, df["log2_fold_change"].to_numpy(dtype=float), colors=colors, linewidth=0.85, alpha=0.7)

    sig_rows = df.loc[sig_mask]
    if not sig_rows.empty:
        sig_colors = [MUTATION_COLORS.get(base, "#777777") for base in sig_rows["seq_alt"]]
        ax.scatter(sig_rows["_x"], sig_rows["log2_fold_change"], s=16, c=sig_colors, zorder=3, alpha=0.95)

    ax.axhline(0.0, color="#666666", lw=1.0)
    ax.axvline(0.0, color="#999999", lw=1.0, ls="--")
    ro_left = min(float(readout_rel_start), float(readout_rel_end))
    ro_right = max(float(readout_rel_start), float(readout_rel_end))
    ax.axvspan(ro_left, ro_right, color="#1f77b4", alpha=0.12, zorder=0)

    # UTR highlight on the main panel, clipped to the visible x-range.
    if highlight_utr and utr_region is not None:
        utr_left, utr_right = interval_to_rel(utr_region.interval, tss_1based, strand)
        if not (utr_right < xmin or utr_left > xmax):
            ax.axvspan(max(utr_left, xmin), min(utr_right, xmax), color="#2ca25f", alpha=0.12, zorder=0)

    ax.grid(axis="y", alpha=0.25)

    if region_entries:
        assert ax_regions is not None
        region_fill_colors = ["#7aa6c2", "#8ab17d", "#d9a66b", "#b48ead", "#9c755f"]
        for index, (_, left, right) in enumerate(region_entries):
            color = region_fill_colors[index % len(region_fill_colors)]
            ax.axvspan(left, right, color=color, alpha=0.12, zorder=0)

        ax_regions.set_ylim(0.0, 3.0)
        ax_regions.set_yticks([])
        ax_regions.spines["left"].set_visible(False)
        ax_regions.spines["right"].set_visible(False)
        ax_regions.spines["top"].set_visible(False)
        ax_regions.spines["bottom"].set_color("#bdbdbd")
        ax_regions.tick_params(axis="x", labelsize=9)
        ax_regions.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
        ax_regions.set_xlabel("Position relative to TSS (bp)")

        for index, (name, left, right) in enumerate(region_entries):
            row = index % 5
            y_base = 2.75 - row * 0.5
            bar_height = 0.16
            color = region_fill_colors[index % len(region_fill_colors)]
            ax_regions.add_patch(
                Rectangle((left, y_base), right - left, bar_height, facecolor=color, edgecolor=color, alpha=0.55)
            )
            ax_regions.text(
                (left + right) / 2.0,
                y_base - 0.05,
                name,
                ha="center",
                va="top",
                fontsize=6.5,
                rotation=0,
                color="#303030",
                clip_on=False,
            )

    ax.set_xlim(xmin, xmax)
    ax.set_ylabel("Log2 fold change")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    if ax_regions is None:
        ax.set_xlabel("Position relative to TSS (bp)")
    else:
        ax.tick_params(labelbottom=False)

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


def prompt_tf_regions_from_ensembl(files_root: Path, mut_interval: GenomicInterval) -> TFSelection:
    heart_xlsx = resolve_heart_tf_path(files_root)

    print(
        "Querying Ensembl REST motif annotations for interval "
        f"{mut_interval.chrom}:{mut_interval.start:,}-{mut_interval.end:,} ..."
    )
    all_regions = fetch_ensembl_tf_regions(mut_interval)
    if not all_regions:
        print("No TF motif intervals returned by Ensembl for this interval.")
        return TFSelection(regions=[], ordered_names=[])
    if not heart_xlsx.exists():
        print(f"Heart TF list not found at {heart_xlsx}; using unfiltered Ensembl TF ranking.")
        return select_tf_regions_by_frequency(all_regions, source_label="Ensembl motif hits")
    return select_tf_regions_by_heart_order(all_regions, heart_xlsx)


def prompt_tf_regions_from_jaspar(files_root: Path, mut_interval: GenomicInterval) -> TFSelection:
    heart_xlsx = resolve_heart_tf_path(files_root)
    heart_tf_names: set[str] | None = None
    if heart_xlsx.exists():
        heart_tf_names = load_heart_tf_names(heart_xlsx)
        if not heart_tf_names:
            print("No TF names found in HeartTSList.xlsx; switching to unfiltered JASPAR TF ranking.")
            heart_tf_names = None
    else:
        print(f"Heart TF list not found at {heart_xlsx}; using unfiltered JASPAR TF ranking.")

    print(
        "Scanning mutagenesis interval with JASPAR motifs: "
        f"{mut_interval.chrom}:{mut_interval.start:,}-{mut_interval.end:,}"
    )
    all_regions = fetch_jaspar_tf_regions(mut_interval, heart_tf_names)
    if not all_regions:
        print("No JASPAR motif intervals returned for this window.")
        return TFSelection(regions=[], ordered_names=[])

    if not heart_xlsx.exists() or not heart_tf_names:
        return select_tf_regions_by_frequency(all_regions, source_label="JASPAR motif hits")

    return select_tf_regions_by_heart_order(all_regions, heart_xlsx)


def prompt_tf_regions_from_csv(files_root: Path, chrom: str) -> TFSelection:
    tf_dir = files_root / "Transcription Factors"
    heart_xlsx = resolve_heart_tf_path(files_root)

    if not tf_dir.exists():
        raise FileNotFoundError(f"TF folder not found: {tf_dir}")

    print(f"Upload your TF CSV to: {tf_dir}")
    available = sorted(path.name for path in tf_dir.glob("*.csv"))
    if available:
        print("Available CSV files:")
        for name in available:
            print(f"  - {name}")

    while True:
        tf_filename = prompt_nonempty("TF CSV filename in 'Transcription Factors': ")
        tf_file = tf_dir / tf_filename
        if tf_file.exists() and tf_file.is_file():
            break
        print(f"File not found: {tf_file}")

    all_regions = load_tf_regions_from_file(tf_file, expected_chrom=chrom)
    if not all_regions:
        print("No TF intervals found on the target chromosome in that file.")
        return TFSelection(regions=[], ordered_names=[])

    if not heart_xlsx.exists():
        print(f"Heart TF list not found at {heart_xlsx}; using unfiltered CSV TF ranking.")
        return select_tf_regions_by_frequency(all_regions, source_label="CSV TF intervals")

    return select_tf_regions_by_heart_order(all_regions, heart_xlsx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive saturation mutagenesis runner.")
    parser.add_argument("--api-key", default=None, help="Optional AlphaGenome API key. Falls back to ALPHAGENOME_API_KEY.")
    parser.add_argument("--output-root", default="Outputs", help="Root output directory (default: Outputs)")
    parser.add_argument(
        "--assets-root",
        default="Files",
        help="Folder containing static input assets like TF CSVs and HeartTSList.xlsx (default: Files)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\nInteractive TSS-centered saturation mutagenesis\n")

    gene = prompt_nonempty("Gene symbol (e.g., NPPB): ").upper()
    ann: GeneAnnotation | None = None
    try:
        ann = lookup_gene_annotation_with_ensembl_rest(gene)
        print("\nAuto-annotation from Ensembl REST:")
    except Exception as rest_exc:
        print(f"\nEnsembl REST lookup failed for {gene}: {rest_exc}")
        print("Trying pybiomart fallback...")
        try:
            ann = lookup_gene_annotation_with_biomart(gene)
            print("\nAuto-annotation from pybiomart fallback:")
        except Exception as biomart_exc:
            print(f"pybiomart fallback failed for {gene}: {biomart_exc}")

    if ann is not None:
        utr_interval = ann.utr_interval
        strand = ann.strand
        tss_1based = ann.tss_1based
        print(f"  Source: {ann.source}")
        print(f"  Transcript: {ann.transcript_id}")
        print(f"  Strand: {strand}")
        print(f"  TSS: {ann.chrom}:{tss_1based:,}")
        print(f"  5' UTR: {utr_interval.chrom}:{utr_interval.start:,}-{utr_interval.end:,}")
    else:
        print("Falling back to manual UTR entry.")
        utr_text = prompt_nonempty("5' UTR interval (e.g., chr1:11,857,464-11,857,654(+)): ")
        utr_interval, parsed_strand = parse_interval(utr_text)
        strand = parsed_strand if parsed_strand in {"+", "-"} else prompt_strand("Gene strand (+ or -): ")
        tss_1based = tss_from_utr(utr_interval, strand)

    upstream = prompt_positive_int("How many bp upstream of TSS for mutagenesis? ")
    downstream = prompt_positive_int("How many bp downstream of TSS for mutagenesis? ")

    mut_interval = mutagenesis_interval(
        chrom=utr_interval.chrom,
        tss_1based=tss_1based,
        strand=strand,
        upstream=upstream,
        downstream=downstream,
    )

    normalize_magnitude = prompt_yes_no(
        "Normalize scores by readout length for cross-run comparability?",
        default_yes=True,
    )
    position_aggregation = "mean" if normalize_magnitude else "sum"

    dataset_key, dataset_label, ontology_terms, post_filter_keywords = prompt_dataset_choice()
    output_label, output_types, output_filter = prompt_output_choice()

    highlight_utr = prompt_yes_no("Highlight UTR on the plot?", default_yes=True)
    highlight_tfs = prompt_yes_no("Highlight transcription factors on the plot?", default_yes=True)

    tf_selection = TFSelection(regions=[], ordered_names=[])
    tf_include_names: list[str] | None = None
    if highlight_tfs:
        tf_include_names = prompt_tf_include_names()
        try:
            tf_selection = prompt_tf_regions_from_jaspar(files_root=Path(args.assets_root), mut_interval=mut_interval)
        except Exception as jaspar_exc:
            print(f"\nJASPAR TF scan failed: {jaspar_exc}")

        if not tf_selection.regions:
            print("\nJASPAR did not return usable TF highlights for this window.")
            if prompt_yes_no("Try Ensembl motif overlaps as fallback?", default_yes=True):
                tf_selection = prompt_tf_regions_from_ensembl(files_root=Path(args.assets_root), mut_interval=mut_interval)
        if not tf_selection.regions:
            if prompt_yes_no("Fall back to TF CSV input for highlights?", default_yes=True):
                tf_selection = prompt_tf_regions_from_csv(files_root=Path(args.assets_root), chrom=utr_interval.chrom)
        tf_selection = filter_tf_selection_by_include_names(tf_selection, tf_include_names)
        print(f"Parsed TF intervals: {len(tf_selection.regions)}")

    readout_rel_start, readout_rel_end = interval_to_rel(mut_interval, tss_1based, strand)

    print("\nRun summary:")
    print(f"  Gene: {gene}")
    print(f"  Strand: {strand}")
    print(f"  TSS: {utr_interval.chrom}:{tss_1based:,}")
    print(f"  5' UTR: {utr_interval.chrom}:{utr_interval.start:,}-{utr_interval.end:,}")
    print(f"  Readout (promoter-wide): {mut_interval.chrom}:{mut_interval.start:,}-{mut_interval.end:,}")
    print(f"  Mutagenesis interval: {mut_interval.chrom}:{mut_interval.start:,}-{mut_interval.end:,}")
    print(f"  Relative window: -{upstream}/+{downstream}")
    print(f"  Position aggregation: {position_aggregation}")
    print(f"  Dataset filter: {dataset_label}")
    print(f"  Output preset: {output_label}")
    print(f"  Output types: {', '.join(output_types)}")
    print(f"  Output filter: {output_filter if output_filter else 'none'}")
    print("  Track grouping: biosample")
    if ontology_terms is None:
        print("  Ontology terms: none")
    else:
        print(f"  Ontology terms: {', '.join(ontology_terms)}")
    if post_filter_keywords:
        print(f"  Strict biosample keywords: {', '.join(post_filter_keywords)}")
    if tf_include_names:
        print(f"  TF include list: {', '.join(tf_include_names)}")
    if not prompt_yes_no("Proceed with this run?", default_yes=True):
        print("Run cancelled by user before scoring.")
        return

    output_dir = Path(args.output_root) / gene
    plots_dir = output_dir / "Plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    prefix_base = f"{gene.lower()}_tss_{tss_1based}_{upstream}up_{downstream}down"
    prefix = choose_unique_run_prefix(output_dir, prefix_base)
    if prefix != prefix_base:
        print(f"Existing output files detected. Using new run prefix: {prefix}")

    context_fasta = output_dir / f"{prefix}_context_150kb_hg38.fa"
    window_fasta = output_dir / f"{prefix}_window.fa"
    table_csv = output_dir / f"{prefix}_variant_effects.csv"
    raw_plot = plots_dir / f"{prefix}_raw.png"
    style_plot = plots_dir / f"{prefix}_style_p_lt_0.05_only.png"
    metadata_summary_csv = output_dir / f"{prefix}_metadata_summary.csv"

    api_key = args.api_key or os.getenv("ALPHAGENOME_API_KEY")
    if not api_key:
        api_key = prompt_nonempty("AlphaGenome API key: ")

    cmd = [
        sys.executable,
        "tss_saturation_pipeline.py",
        "--tss",
        f"{utr_interval.chrom}:{tss_1based}",
        "--strand",
        strand,
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
        str(mut_interval.start),
        "--readout-end-genomic",
        str(mut_interval.end),
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

    print("\nRunning saturation mutagenesis...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    utr_region = Region(name="UTR", interval=utr_interval)
    subtitle = (
        f"mutagenesis window: -{upstream}/+{downstream} | "
        f"readout (promoter): {mut_interval.chrom}:{mut_interval.start:,}-{mut_interval.end:,}"
    )

    print("\nGenerating DSP-style plot...")
    plot_dsp_style(
        csv_path=table_csv,
        out_path=style_plot,
        tss_1based=tss_1based,
        strand=strand,
        xmin=-float(upstream),
        xmax=float(downstream),
        title=f"{gene} TSS-centered SNV effects",
        subtitle=subtitle,
        readout_rel_start=readout_rel_start,
        readout_rel_end=readout_rel_end,
        highlight_utr=highlight_utr,
        utr_region=utr_region,
        tf_regions=tf_selection.regions,
    )

    run_summary = [
        "Run completion summary:",
        f"  Completed at: {datetime.now().isoformat(timespec='seconds')}",
        f"  Gene: {gene}",
        f"  Strand: {strand}",
        f"  TSS: {utr_interval.chrom}:{tss_1based:,}",
        f"  5' UTR: {format_interval(utr_interval)}",
        f"  Readout (promoter-wide): {format_interval(mut_interval)}",
        f"  Relative window: -{upstream}/+{downstream}",
        f"  Position aggregation: {position_aggregation}",
        f"  Dataset filter: {dataset_label}",
        f"  Output preset: {output_label}",
        f"  Output types: {', '.join(output_types)}",
        "  Track grouping: biosample",
        f"  Ontology terms: {', '.join(ontology_terms) if ontology_terms else 'none'}",
        f"  Strict biosample keywords: {', '.join(post_filter_keywords) if post_filter_keywords else 'none'}",
        f"  TF include list: {', '.join(tf_include_names) if tf_include_names else 'all selected'}",
        f"  Metadata summary: {metadata_summary_csv}",
        f"  Data table: {table_csv}",
        f"  Raw plot: {raw_plot}",
        f"  Styled plot: {style_plot}",
    ]
    run_summary.extend(summarize_tf_intervals(tf_selection))

    summary_text = "\n".join(run_summary)
    summary_file = output_dir / f"{prefix}_run_summary.txt"
    summary_file.write_text(summary_text + "\n", encoding="utf-8")

    print("\nDone.")
    print(f"TSS used: {utr_interval.chrom}:{tss_1based} ({strand} strand)")
    print(f"Data table: {table_csv}")
    print(f"Raw plot: {raw_plot}")
    print(f"Styled plot: {style_plot}")
    print(f"Summary file: {summary_file}")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
