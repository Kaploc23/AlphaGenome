#!/usr/bin/env python3
"""Run TSS-centered SNV saturation mutagenesis with large genomic context.

Pipeline:
1) Fetch hg38 genomic sequence around a TSS with large flanks.
2) Build all SNVs in a local window around the TSS (e.g., -500..+50).
3) Score WT and all mutants with AlphaGenome.
4) Write a scored table and a color-coded thin-bar plot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from alphagenome.data import track_data
from alphagenome.models import dna_client, dna_model, dna_output
from scipy.stats import norm
from tqdm import tqdm

BASES = ("A", "C", "G", "T")
SUPPORTED_CONTEXT_LENGTHS = (16384, 131072, 524288, 1048576)
CARDIAC_ONTOLOGY_TERMS = (
  "UBERON:0000948",  # heart
  "CL:0000746",      # cardiac muscle cell
)

# Colors requested by user for mutant base (ALT).
ALT_COLORS = {
  "A": "#2ca25f",  # green
  "T": "#2b8cbe",  # blue
  "C": "#de2d26",  # red
  "G": "#f39c12",  # orange
}
ALT_X_OFFSET = {
  "A": -0.24,
  "C": -0.08,
  "G": 0.08,
  "T": 0.24,
}

DNA_TRACK_COLORS = {
  "A": "#2ca25f",
  "C": "#2b8cbe",
  "G": "#f39c12",
  "T": "#de2d26",
}


def parse_tss_coordinate(coord: str) -> tuple[str, int]:
  raw = coord.strip()
  if ":" not in raw:
    raise ValueError("TSS coordinate must be like chr11:47,352,703")
  chrom, pos = raw.split(":", 1)
  pos_1based = int(pos.replace(",", ""))
  if pos_1based < 1:
    raise ValueError("TSS coordinate must be >= 1")
  return chrom, pos_1based


def parse_genomic_interval(interval: str) -> tuple[str, int, int]:
  raw = interval.strip()
  if ":" not in raw or "-" not in raw:
    raise ValueError("Genomic interval must be like chr1:11,857,464-11,857,654")
  chrom, coords = raw.split(":", 1)
  start_s, end_s = coords.split("-", 1)
  start_1based = int(start_s.replace(",", ""))
  end_1based = int(end_s.replace(",", ""))
  if start_1based < 1 or end_1based < 1:
    raise ValueError("Genomic interval positions must be >= 1")
  if end_1based < start_1based:
    start_1based, end_1based = end_1based, start_1based
  return chrom, start_1based, end_1based


def choose_context_length(required_length: int) -> int:
  for value in SUPPORTED_CONTEXT_LENGTHS:
    if value >= required_length:
      return value
  raise ValueError(
      f"Required length {required_length} exceeds max supported context "
      f"length {SUPPORTED_CONTEXT_LENGTHS[-1]}"
  )


def fetch_hg38_sequence(chrom: str, start_1based: int, end_1based: int) -> str:
  # UCSC API uses 0-based half-open intervals.
  url = (
      "https://api.genome.ucsc.edu/getData/sequence"
      f"?genome=hg38;chrom={chrom};start={start_1based - 1};end={end_1based}"
  )
  with urllib.request.urlopen(url, timeout=90) as response:
    payload = json.loads(response.read().decode("utf-8"))
  seq = payload.get("dna", "").upper()
  if not seq:
    raise RuntimeError("No DNA sequence returned by UCSC API.")
  expected = end_1based - start_1based + 1
  if len(seq) != expected:
    raise RuntimeError(f"Fetched length {len(seq)} does not match expected {expected}.")
  return seq


def write_fasta(path: Path, header: str, sequence: str, width: int = 80) -> None:
  with path.open("w", encoding="utf-8") as handle:
    handle.write(f">{header}\n")
    for i in range(0, len(sequence), width):
      handle.write(sequence[i : i + width] + "\n")


def read_first_fasta(path: Path) -> str:
  sequence_parts: list[str] = []
  with path.open("r", encoding="utf-8") as handle:
    for raw in handle:
      line = raw.strip()
      if not line:
        continue
      if line.startswith(">"):
        continue
      sequence_parts.append(line)
  if not sequence_parts:
    raise ValueError(f"No sequence found in FASTA: {path}")
  return "".join(sequence_parts).upper()


def embed_sequence(sequence: str, context_length: int, flank_base: str = "N") -> tuple[str, int, int]:
  if len(sequence) > context_length:
    raise ValueError("Sequence is longer than the model context length.")
  left = (context_length - len(sequence)) // 2
  right = context_length - len(sequence) - left
  embedded = flank_base * left + sequence + flank_base * right
  return embedded, left, left + len(sequence)


def reduce_values(values: np.ndarray, mode: str) -> float:
  if mode == "sum":
    return float(np.sum(values))
  if mode == "mean":
    return float(np.mean(values))
  if mode == "max":
    return float(np.max(values))
  raise ValueError(mode)


def normalize_text_series(series: pd.Series) -> pd.Series:
  return series.fillna('').astype(str).str.strip().str.lower()


def build_metadata_keyword_mask(metadata: pd.DataFrame, keywords: list[str]) -> np.ndarray:
  if not keywords:
    return np.ones(len(metadata), dtype=bool)

  normalized_keywords = [keyword.strip().lower() for keyword in keywords if keyword.strip()]
  if not normalized_keywords:
    return np.ones(len(metadata), dtype=bool)

  candidate_columns = [
      'biosample_name',
      'gtex_tissue',
      'name',
      'cell_type',
      'biosample_type',
      'Assay title',
  ]
  present_columns = [col for col in candidate_columns if col in metadata.columns]
  if not present_columns:
    return np.zeros(len(metadata), dtype=bool)

  mask = np.zeros(len(metadata), dtype=bool)
  for col in present_columns:
    values = normalize_text_series(metadata[col])
    for keyword in normalized_keywords:
      mask |= values.str.contains(keyword, regex=False).to_numpy(dtype=bool)
  return mask


def filter_track_data_by_metadata(
    tdata: track_data.TrackData,
    ontology_curies: list[str] | None,
    biosample_keywords: list[str] | None,
) -> track_data.TrackData | None:
  metadata = tdata.metadata
  ontology_mask = np.ones(len(metadata), dtype=bool)
  if ontology_curies and 'ontology_curie' in metadata.columns:
    ontology_mask = metadata['ontology_curie'].isin(ontology_curies).to_numpy(dtype=bool)

  keyword_mask = np.ones(len(metadata), dtype=bool)
  if biosample_keywords:
    keyword_mask = build_metadata_keyword_mask(metadata, biosample_keywords)

  candidate_masks = []
  if ontology_curies and biosample_keywords:
    candidate_masks.extend([
        ontology_mask & keyword_mask,
        ontology_mask,
        keyword_mask,
    ])
  elif ontology_curies:
    candidate_masks.append(ontology_mask)
  elif biosample_keywords:
    candidate_masks.append(keyword_mask)
  else:
    candidate_masks.append(np.ones(len(metadata), dtype=bool))

  chosen_mask = None
  for mask in candidate_masks:
    if np.any(mask):
      chosen_mask = mask
      break

  if chosen_mask is None:
    return None

  filtered = tdata.filter_tracks(chosen_mask)
  if filtered.num_tracks == 0:
    return None
  return filtered


def filter_output_by_metadata(
    output: dna_output.Output,
    ontology_curies: list[str] | None,
    biosample_keywords: list[str] | None,
) -> dna_output.Output:
  def _filter(
      tdata: track_data.TrackData,
      output_type: dna_output.OutputType,
  ) -> track_data.TrackData | None:
    del output_type
    return filter_track_data_by_metadata(tdata, ontology_curies, biosample_keywords)

  return output.map_track_data(_filter)


def track_group_labels(metadata: pd.DataFrame) -> pd.Series:
  if 'biosample_name' in metadata.columns:
    biosample = metadata['biosample_name'].fillna('').astype(str).str.strip()
    if biosample.ne('').any():
      return biosample.where(biosample.ne(''), metadata.get('name', biosample).astype(str))
  if 'gtex_tissue' in metadata.columns:
    tissue = metadata['gtex_tissue'].fillna('').astype(str).str.strip()
    if tissue.ne('').any():
      return tissue.where(tissue.ne(''), metadata.get('name', tissue).astype(str))
  return metadata['name'].fillna('').astype(str)


def aggregate_track_values(
    per_track: np.ndarray,
    metadata: pd.DataFrame,
    track_grouping: str,
) -> np.ndarray:
  if track_grouping == 'tracks':
    return np.asarray(per_track, dtype=np.float64)
  if track_grouping != 'biosample':
    raise ValueError(track_grouping)

  labels = track_group_labels(metadata)
  grouped: dict[str, list[float]] = {}
  for value, label in zip(np.asarray(per_track, dtype=np.float64), labels, strict=True):
    key = str(label).strip() or 'unknown'
    grouped.setdefault(key, []).append(float(value))
  return np.asarray([float(np.mean(values)) for values in grouped.values()], dtype=np.float64)


def collect_output_metadata(output: dna_output.Output) -> pd.DataFrame:
  frames: list[pd.DataFrame] = []
  for output_type in dna_output.OutputType:
    track = output.get(output_type)
    if track is None:
      continue
    frame = track.metadata.copy()
    frame['output_type'] = output_type.name
    frames.append(frame)
  if not frames:
    return pd.DataFrame()
  return pd.concat(frames, ignore_index=True)


def summarize_output_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
  if metadata.empty:
    return pd.DataFrame(columns=['output_type', 'biosample_group', 'track_count'])

  labels = track_group_labels(metadata)
  frame = metadata.copy()
  frame['biosample_group'] = labels
  summary = (
      frame.groupby(['output_type', 'biosample_group'], dropna=False)
      .size()
      .reset_index(name='track_count')
      .sort_values(['output_type', 'track_count', 'biosample_group'], ascending=[True, False, True])
      .reset_index(drop=True)
  )
  return summary


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
  n = len(pvals)
  order = np.argsort(pvals)
  ranked = pvals[order]
  adjusted = np.empty(n, dtype=float)
  prev = 1.0
  for i in range(n - 1, -1, -1):
    rank = i + 1
    val = min(prev, ranked[i] * n / rank)
    adjusted[i] = val
    prev = val
  out = np.empty(n, dtype=float)
  out[order] = np.clip(adjusted, 0.0, 1.0)
  return out


def robust_pvalues(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  med = float(np.median(delta))
  mad = float(np.median(np.abs(delta - med)))
  scale = 1.4826 * mad
  if not np.isfinite(scale) or scale <= 1e-12:
    scale = float(np.std(delta, ddof=1)) if len(delta) > 1 else 1.0
  if not np.isfinite(scale) or scale <= 1e-12:
    scale = 1.0
  z = (delta - med) / scale
  pvals = 2.0 * norm.sf(np.abs(z))
  pvals_adj = bh_adjust(pvals)
  return z, pvals, pvals_adj


def score_output(
    output: dna_output.Output,
    output_types: list[dna_output.OutputType],
    promoter_start: int,
    promoter_end: int,
    position_aggregation: str,
    track_aggregation: str,
    output_aggregation: str,
    track_grouping: str,
) -> tuple[dict[str, float], float]:
  per_output: dict[str, float] = {}
  for output_type in output_types:
    track = output.get(output_type)
    if track is None:
      continue
    resolution = int(track.resolution)
    start_bin = promoter_start // resolution
    end_bin = int(np.ceil(promoter_end / resolution))
    window = track.values[start_bin:end_bin, ...]
    if window.ndim == 1:
      window = window[np.newaxis, ...]
    flattened = window.reshape(-1, window.shape[-1])
    if position_aggregation == "sum":
      per_track = flattened.sum(axis=0)
    elif position_aggregation == "mean":
      per_track = flattened.mean(axis=0)
    elif position_aggregation == "max":
      per_track = flattened.max(axis=0)
    else:
      raise ValueError(position_aggregation)
    grouped_values = aggregate_track_values(per_track, track.metadata, track_grouping)
    per_output[output_type.name] = reduce_values(grouped_values, track_aggregation)

  if not per_output:
    raise ValueError("No requested outputs were returned by AlphaGenome.")
  combined = reduce_values(np.asarray(list(per_output.values()), dtype=np.float64), output_aggregation)
  return per_output, combined


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="TSS-centered saturation mutagenesis with AlphaGenome.")
  parser.add_argument("--tss", required=True, help="TSS coordinate like chr11:47,352,703")
  parser.add_argument("--strand", choices=["+", "-"], default="+", help="Gene strand for relative position annotation.")
  parser.add_argument("--upstream", type=int, default=500, help="Mutagenesis window upstream bp (default 500).")
  parser.add_argument("--downstream", type=int, default=50, help="Mutagenesis window downstream bp (default 50).")
  parser.add_argument("--flank", type=int, default=150000, help="Context flank on each side of TSS (default 150000).")
  parser.add_argument("--api-key", default=None, help="AlphaGenome API key; defaults to ALPHAGENOME_API_KEY.")
  parser.add_argument("--organism", default="HOMO_SAPIENS", choices=["HOMO_SAPIENS", "MUS_MUSCULUS"])
  parser.add_argument("--output-types", nargs="+", default=["CAGE"])
  parser.add_argument("--ontology-terms", nargs="+", default=None, help="Ontology term CURIEs (e.g. UBERON:0000948 CL:0000746).")
  parser.add_argument("--cardiac-only", action="store_true", help="Restrict to cardiac ontology terms known to be supported.")
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--max-workers", type=int, default=1)
  parser.add_argument(
      "--progress-log-every",
      type=int,
      default=1,
      help="Print an explicit percent-complete log every N percent (default 1).",
  )
  parser.add_argument("--position-aggregation", choices=["sum", "mean", "max"], default="sum")
  parser.add_argument("--track-aggregation", choices=["sum", "mean", "max"], default="mean")
  parser.add_argument("--output-aggregation", choices=["sum", "mean", "max"], default="mean")
  parser.add_argument(
      "--track-grouping",
      choices=["tracks", "biosample"],
      default="tracks",
      help="Aggregate equally across raw tracks or across grouped biosamples.",
  )
  parser.add_argument(
      "--post-filter-ontology-curies",
      nargs="+",
      default=None,
      help="Optional strict post-filter ontology curies applied to returned track metadata.",
  )
  parser.add_argument(
      "--post-filter-biosample-keywords",
      nargs="+",
      default=None,
      help="Optional case-insensitive biosample keywords applied to returned track metadata.",
  )
  parser.add_argument(
      "--metadata-summary-out",
      default=None,
      help="Optional CSV path for a summary of returned tracks grouped by biosample after post-filtering.",
  )
  parser.add_argument(
      "--context-fasta-in",
      default=None,
      help="Optional existing context FASTA path to use instead of fetching from UCSC.",
  )
  parser.add_argument(
      "--window-fasta-in",
      default=None,
      help="Optional existing window FASTA path; if omitted, window is sliced from context.",
  )
  parser.add_argument("--context-fasta-out", default="Files/tss_context_150kb_hg38.fa")
  parser.add_argument("--window-fasta-out", default="Files/tss_window_500up_50down_hg38.fa")
  parser.add_argument("--outfile", default="Files/tss_chr11_47352703_saturation_scores.csv")
  parser.add_argument("--plot-out", default="Files/tss_chr11_47352703_saturation_plot_three_mut_bars.png")
  parser.add_argument(
      "--readout-start-genomic",
      type=int,
      default=None,
      help="Optional 1-based genomic readout start used for scoring aggregation.",
  )
  parser.add_argument(
      "--readout-end-genomic",
      type=int,
      default=None,
      help="Optional 1-based genomic readout end used for scoring aggregation.",
  )
  parser.add_argument(
      "--readout-interval",
      default=None,
      help="Optional genomic readout interval like chr1:11,857,464-11,857,654.",
  )
  return parser.parse_args()


def resolve_ontology_terms(args: argparse.Namespace) -> list[str] | None:
  terms: list[str] = []
  if args.ontology_terms:
    terms.extend(t for t in args.ontology_terms if str(t).strip())
  if args.cardiac_only:
    terms.extend(CARDIAC_ONTOLOGY_TERMS)
  if not terms:
    return None
  return list(dict.fromkeys(terms))


def draw_sequence_track(
    ax: plt.Axes,
    window_seq: str,
    x_positions: np.ndarray,
    strand: str,
) -> None:
  ax.set_ylim(0, 1)
  ax.set_yticks([])
  ax.spines["left"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["top"].set_visible(False)
  ax.spines["bottom"].set_color("#bdbdbd")

  for x_rel, base in zip(x_positions, window_seq, strict=True):
    ax.add_patch(
        plt.Rectangle(
            (x_rel - 0.5, 0.15),
            1.0,
            0.7,
            facecolor=DNA_TRACK_COLORS.get(base, "#cccccc"),
            edgecolor="none",
            alpha=0.18,
        )
    )

  label_span = 120
  for x_rel, base in zip(x_positions, window_seq, strict=True):
    abs_x = abs(int(x_rel))
    if abs_x <= 25 or (abs_x <= label_span and abs_x % 5 == 0) or abs_x % 25 == 0:
      ax.text(
          x_rel,
          0.52,
          base,
          ha="center",
          va="center",
          fontsize=6 if abs_x > 25 else 7,
          fontweight="bold",
          color=DNA_TRACK_COLORS.get(base, "#444444"),
      )

  ax.axvline(0, color="#6a6a6a", lw=1.0, ls="--")
  ax.text(0, 0.92, "TSS", ha="center", va="top", fontsize=10, fontweight="bold")
  ax.text(
      0.01,
      0.05,
      f"strand {strand}",
      transform=ax.transAxes,
      ha="left",
      va="bottom",
      fontsize=9,
      color="#555555",
  )
  ax.set_xlim(float(x_positions.min()) - 0.5, float(x_positions.max()) + 0.5)
  ax.set_xlabel("Position relative to TSS (bp)")
  ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))


def build_track_positions(mut_start: int, mut_end: int, tss: int, strand: str) -> np.ndarray:
  if strand == "+":
    return np.arange(mut_start - tss, mut_end - tss + 1)
  # Keep genomic order while expressing positions in transcript-relative coordinates.
  return np.arange(tss - mut_start, tss - mut_end - 1, -1)


def main() -> None:
  args = parse_args()

  api_key = args.api_key or os.getenv("ALPHAGENOME_API_KEY")
  if not api_key:
    raise ValueError("AlphaGenome API key not provided. Use --api-key or set ALPHAGENOME_API_KEY.")

  chrom, tss = parse_tss_coordinate(args.tss)
  if args.strand == "+":
    mut_start = tss - args.upstream
    mut_end = tss + args.downstream
  else:
    # On the minus strand, upstream is toward increasing genomic coordinates.
    mut_start = tss - args.downstream
    mut_end = tss + args.upstream
  ctx_start = tss - args.flank
  ctx_end = tss + args.flank

  if mut_start < 1 or ctx_start < 1:
    raise ValueError("Computed interval starts before position 1; adjust upstream/flank.")

  readout_chrom = chrom
  readout_start = ctx_start
  readout_end = ctx_end
  if args.readout_interval:
    readout_chrom, readout_start, readout_end = parse_genomic_interval(args.readout_interval)
  elif args.readout_start_genomic is not None or args.readout_end_genomic is not None:
    if args.readout_start_genomic is None or args.readout_end_genomic is None:
      raise ValueError("Both --readout-start-genomic and --readout-end-genomic are required together.")
    readout_start = int(args.readout_start_genomic)
    readout_end = int(args.readout_end_genomic)
    if readout_end < readout_start:
      readout_start, readout_end = readout_end, readout_start

  if readout_chrom != chrom:
    raise ValueError(f"Readout chromosome {readout_chrom} does not match TSS chromosome {chrom}.")
  if readout_start < ctx_start or readout_end > ctx_end:
    raise ValueError(
        f"Readout interval {chrom}:{readout_start}-{readout_end} must be within context {chrom}:{ctx_start}-{ctx_end}."
    )

  required_context = ctx_end - ctx_start + 1
  context_length = choose_context_length(required_context)

  if args.context_fasta_in:
    wt_context_seq = read_first_fasta(Path(args.context_fasta_in))
    if len(wt_context_seq) != required_context:
      raise ValueError(
          f"Context FASTA length {len(wt_context_seq)} does not match expected {required_context}."
      )
    print(f"Loaded context FASTA input: {args.context_fasta_in}")
  else:
    wt_context_seq = fetch_hg38_sequence(chrom, ctx_start, ctx_end)
  window_offset_start = mut_start - ctx_start
  window_offset_end = mut_end - ctx_start
  if args.window_fasta_in:
    window_seq = read_first_fasta(Path(args.window_fasta_in))
    expected_window = window_offset_end - window_offset_start + 1
    if len(window_seq) != expected_window:
      raise ValueError(
          f"Window FASTA length {len(window_seq)} does not match expected {expected_window}."
      )
    print(f"Loaded window FASTA input: {args.window_fasta_in}")
  else:
    window_seq = wt_context_seq[window_offset_start : window_offset_end + 1]

  context_header = f"{chrom}:{ctx_start}-{ctx_end}_tss_{tss}_hg38"
  window_header = f"{chrom}:{mut_start}-{mut_end}_tss_{tss}_hg38"
  write_fasta(Path(args.context_fasta_out), context_header, wt_context_seq)
  write_fasta(Path(args.window_fasta_out), window_header, window_seq)

  ontology_terms = resolve_ontology_terms(args)
  effective_ontology_terms = ontology_terms
  effective_post_filter_ontology_curies = args.post_filter_ontology_curies
  effective_post_filter_biosample_keywords = args.post_filter_biosample_keywords
  if ontology_terms is None:
    print("Ontology request filters: none (all matching datasets).")
  else:
    print("Ontology request filters:", ", ".join(ontology_terms))
  if effective_post_filter_ontology_curies:
    print("Ontology post-filters (local):", ", ".join(effective_post_filter_ontology_curies))

  model = dna_client.create(api_key)
  organism = dna_model.Organism[args.organism]
  output_types = [dna_output.OutputType[name.upper()] for name in args.output_types]

  wt_embedded, wt_start, wt_end = embed_sequence(wt_context_seq, context_length)
  print("Running wild-type prediction (this can take several minutes for large windows)...", flush=True)
  try:
    wt_output = model.predict_sequence(
        sequence=wt_embedded,
        organism=organism,
        requested_outputs=output_types,
        ontology_terms=effective_ontology_terms,
    )
  except Exception as exc:
    if (
        effective_ontology_terms is not None
        and "Unsupported ontology" in str(exc)
    ):
      print(
          "AlphaGenome rejected at least one ontology CURIE. "
          "Retrying without request-time ontology filtering.",
          flush=True,
      )
      effective_ontology_terms = None
      effective_post_filter_ontology_curies = None
      effective_post_filter_biosample_keywords = None
      wt_output = model.predict_sequence(
          sequence=wt_embedded,
          organism=organism,
          requested_outputs=output_types,
          ontology_terms=effective_ontology_terms,
      )
    else:
      raise
  print("Wild-type prediction complete.", flush=True)
  filtered_wt_output = filter_output_by_metadata(
      wt_output,
      ontology_curies=effective_post_filter_ontology_curies,
      biosample_keywords=effective_post_filter_biosample_keywords,
  )
  readout_start_offset = readout_start - ctx_start
  readout_end_offset = readout_end - ctx_start
  readout_start_embedded = wt_start + readout_start_offset
  readout_end_embedded = wt_start + readout_end_offset + 1
  try:
    _, wt_total = score_output(
        filtered_wt_output,
        output_types,
        promoter_start=readout_start_embedded,
        promoter_end=readout_end_embedded,
        position_aggregation=args.position_aggregation,
        track_aggregation=args.track_aggregation,
        output_aggregation=args.output_aggregation,
        track_grouping=args.track_grouping,
    )
  except ValueError as exc:
    if (
        "No requested outputs were returned by AlphaGenome." in str(exc)
        and (effective_post_filter_ontology_curies or effective_post_filter_biosample_keywords)
    ):
      print(
          "Strict post-filtering removed all requested outputs. "
          "Retrying without post-filter ontology/keyword constraints.",
          flush=True,
      )
      effective_post_filter_ontology_curies = None
      effective_post_filter_biosample_keywords = None
      filtered_wt_output = wt_output
      _, wt_total = score_output(
          filtered_wt_output,
          output_types,
          promoter_start=readout_start_embedded,
          promoter_end=readout_end_embedded,
          position_aggregation=args.position_aggregation,
          track_aggregation=args.track_aggregation,
          output_aggregation=args.output_aggregation,
          track_grouping=args.track_grouping,
      )
    else:
      raise

  filtered_metadata = collect_output_metadata(filtered_wt_output)
  if args.metadata_summary_out is not None:
    summary = summarize_output_metadata(filtered_metadata)
    Path(args.metadata_summary_out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.metadata_summary_out, index=False)
    print(f"Wrote metadata summary: {args.metadata_summary_out}")

  variants: list[dict[str, str | int]] = []
  for i, ref in enumerate(window_seq):
    pos_1based = i + 1
    gpos = mut_start + i
    rel_tss = gpos - tss if args.strand == "+" else tss - gpos
    for alt in BASES:
      if alt == ref:
        continue
      variants.append(
          {
              "variant_id": f"TSS_{chrom}_{tss}_{ref}{pos_1based}{alt}",
              "wt_position_1based": pos_1based,
              "position_rel_tss": rel_tss,
              "chrom": chrom,
              "genomic_position_1based": gpos,
              "strand": args.strand,
              "seq_ref": ref,
              "seq_alt": alt,
              "mutation": f"{ref}{pos_1based}{alt}",
              "genomic_hgvs": f"{chrom}:g.{gpos}{ref}>{alt}",
              "context_offset_0based": window_offset_start + i,
          }
      )

  print(f"Context interval: {chrom}:{ctx_start}-{ctx_end} ({required_context} bp)")
  print(f"Mutagenesis window: {chrom}:{mut_start}-{mut_end} ({len(window_seq)} bp)")
  print(f"Readout window: {chrom}:{readout_start}-{readout_end} ({readout_end - readout_start + 1} bp)")
  print(f"Model context length used: {context_length}")
  print(f"Variants to score: {len(variants)}")
  print(f"Track grouping: {args.track_grouping}")
  if effective_ontology_terms is None and ontology_terms is not None:
    print("Effective ontology request filter: none (fallback after unsupported CURIE)")
  if effective_post_filter_ontology_curies:
    print("Strict post-filter ontology curies:", ", ".join(effective_post_filter_ontology_curies))
  if effective_post_filter_biosample_keywords:
    print("Strict post-filter biosample keywords:", ", ".join(effective_post_filter_biosample_keywords))
  print("Starting mutant batch scoring...", flush=True)

  rows: list[dict[str, str | int | float]] = []
  batch_sequences: list[str] = []
  batch_meta: list[dict[str, str | int]] = []
  progress = tqdm(
      total=len(variants),
      desc="Scoring TSS-window mutants",
      unit="variant",
      dynamic_ncols=True,
      ascii=True,
  )
  progress_step = max(1, int(args.progress_log_every))
  next_progress_pct = progress_step

  def flush_batch() -> None:
    nonlocal batch_sequences, batch_meta, next_progress_pct
    if not batch_sequences:
      return
    outputs = model.predict_sequences(
        sequences=batch_sequences,
        organism=organism,
        requested_outputs=output_types,
        ontology_terms=effective_ontology_terms,
        progress_bar=False,
        max_workers=args.max_workers,
    )
    for meta, output in zip(batch_meta, outputs, strict=True):
      output = filter_output_by_metadata(
        output,
          ontology_curies=effective_post_filter_ontology_curies,
          biosample_keywords=effective_post_filter_biosample_keywords,
      )
      _, mut_total = score_output(
          output,
          output_types,
          promoter_start=readout_start_embedded,
          promoter_end=readout_end_embedded,
          position_aggregation=args.position_aggregation,
          track_aggregation=args.track_aggregation,
          output_aggregation=args.output_aggregation,
        track_grouping=args.track_grouping,
      )
      rows.append(
          {
              **{k: v for k, v in meta.items() if k != "context_offset_0based"},
              "wild_type_productivity": wt_total,
              "mutant_productivity": mut_total,
              "delta_productivity": mut_total - wt_total,
              "log2_fold_change": float(math.log2(mut_total / wt_total)) if wt_total > 0 and mut_total > 0 else float("nan"),
          }
      )
      progress.update(1)
      completed = progress.n
      if progress.total:
        pct_complete = int((completed * 100) / progress.total)
        while pct_complete >= next_progress_pct and next_progress_pct <= 100:
          print(f"Progress: {next_progress_pct}% ({completed}/{progress.total})")
          next_progress_pct += progress_step
    batch_sequences = []
    batch_meta = []

  for meta in variants:
    idx = int(meta["context_offset_0based"])
    alt = str(meta["seq_alt"])
    mut_context = wt_context_seq[:idx] + alt + wt_context_seq[idx + 1 :]
    embedded, _, _ = embed_sequence(mut_context, context_length)
    batch_sequences.append(embedded)
    batch_meta.append(meta)
    if len(batch_sequences) >= args.batch_size:
      flush_batch()

  flush_batch()
  progress.close()

  results = pd.DataFrame(rows)
  if results.empty:
    raise RuntimeError("No scores produced.")

  # Add robust p-values so plotting can show significance-only dots.
  z, pvals, p_adj = robust_pvalues(results["delta_productivity"].to_numpy(dtype=float))
  results["z_robust"] = z
  results["p_value"] = pvals
  results["p_adj_bh"] = p_adj
  results["is_significant_p05"] = results["p_adj_bh"] < 0.05

  results = results.sort_values(by="delta_productivity", key=lambda c: np.abs(c), ascending=False).reset_index(drop=True)
  results.to_csv(args.outfile, index=False)

  plot_df = results.copy()
  plot_df = plot_df[np.isfinite(plot_df["log2_fold_change"])].copy()
  plot_df["color"] = plot_df["seq_alt"].map(ALT_COLORS)
  plot_df["position_rel_tss"] = pd.to_numeric(plot_df["position_rel_tss"], errors="coerce")
  plot_df = plot_df.dropna(subset=["position_rel_tss"])
  plot_df["x_rel"] = plot_df["position_rel_tss"] + plot_df["seq_alt"].map(ALT_X_OFFSET).fillna(0.0)

  fig, ax = plt.subplots(1, 1, figsize=(16, 6), constrained_layout=True)
  ax.vlines(plot_df["x_rel"], 0.0, plot_df["log2_fold_change"], colors=plot_df["color"], linewidth=0.55, alpha=0.8)

  sig_df = plot_df[plot_df["is_significant_p05"].astype(bool)] if "is_significant_p05" in plot_df.columns else plot_df.iloc[0:0]
  if not sig_df.empty:
    ax.scatter(sig_df["x_rel"], sig_df["log2_fold_change"], s=14, c=sig_df["color"], alpha=0.95, linewidths=0)

  ax.axhline(0, color="#555555", lw=0.9)
  ax.axvline(0, color="#6a6a6a", lw=1.0, ls="--", alpha=0.8)
  ax.grid(axis="y", alpha=0.2, linestyle="-")
  ax.set_title(
      f"TSS-centered SNV effects ({chrom}:{tss}, -{args.upstream}/+{args.downstream}, "
      f"{args.flank // 1000} kb flank)"
  )
  ax.set_ylabel("log2 fold change")
  ax.set_xlabel("Position relative to TSS (bp)")
  ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
  for base in ("A", "T", "C", "G"):
    ax.scatter([], [], c=ALT_COLORS[base], s=35, label=f"WT->{base}")
  ax.scatter([], [], c="#111111", s=20, label="dot: p < 0.05")
  ax.legend(frameon=False, ncol=4, loc="upper right")

  fig.savefig(args.plot_out, dpi=220)

  print(f"WT productivity: {wt_total:.6g}")
  print(f"Variants scored: {len(results)}")
  print(f"Wrote context FASTA: {args.context_fasta_out}")
  print(f"Wrote window FASTA: {args.window_fasta_out}")
  print(f"Wrote table: {args.outfile}")
  print(f"Wrote plot: {args.plot_out}")


if __name__ == "__main__":
  main()
