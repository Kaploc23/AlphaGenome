
import json
import numpy as np

# Import public Matrix/Cluster types from your existing package
from .matrix import Matrix, Cluster

ALPHABET = ("A", "C", "G", "T")


def dl_row_to_matrix(r):
    """Convert a joined DL row into a public Matrix.

    Expected keys (from dl_sql backend SELECT):

    - matrix_data (JSON), matrix_type in {'PFM','CWM'}

    - base_id/version (preferred accession) or dl_motif_id fallback

    - various metadata fields (tf_name, tax_id, species, tax_group, etc.)
    """
    # Accession
    if r.get("base_id") and r.get("version"):
        acc = f"{r['base_id']}.{r['version']}"
    elif r.get("member_base_id") and r.get("member_version"):
        acc = f"{r['member_base_id']}.{r['member_version']}"
    else:
        acc = str(r.get("dl_motif_id") or r.get("motif_id"))

    pfm, kind = _pfm_from_json(r["matrix_data"], r.get("matrix_type"))

    meta = {
        "matrix_id": acc,
        "name": r.get("motif_name") or r.get("tf_name") or acc,
        "collection": "DL",
        "version": r.get("version") or r.get("member_version"),
        "species": r.get("tax_id"),
        "species_name": r.get("species"),
        "tax_group": r.get("tax_group") or r.get("cluster_tax_group"),
        "class": r.get("tf_class"),
        "family": r.get("tf_family"),
        "uniprot_id": r.get("uniprot_id"),
        "tf_name": r.get("tf_name") or r.get("member_tf_name"),
        "cell_line": r.get("cell_line"),
        "data_type": r.get("data_type"),
        "model_name": r.get("model_name"),
        "model_source": r.get("model_source"),
        "model_source_id": r.get("model_source_id"),
        "model_source_url": r.get("model_source_url"),
        "num_seqlets": r.get("num_seqlets"),
        "dl_matrix_type": r.get("matrix_type"),
        "payload_kind": kind,  # 'pfm' or 'cwm-as-pfm'
        "alphabet": ALPHABET,
    }

    return Matrix(
        matrix_id=acc,
        name=meta["name"],
        collection="DL",
        pfm=pfm,
        metadata=meta,
    )


def dl_cluster_rows_to_cluster(group):
    """Convert a grouped set of rows into a Cluster object.

    group = {"cluster": representative_row, "members": [rows...]}
    """
    r0 = group["cluster"]
    if r0.get("base_id") and r0.get("version"):
        cid = f"{r0['base_id']}.{r0['version']}"
    else:
        cid = r0.get("cluster_id")
    name = r0.get("tf_name") or r0.get("motif_name") or cid
    method = "DL-PRIMARY"  # replace with real method if/when exposed
    members = [dl_row_to_matrix(r) for r in group["members"]]
    return Cluster(cluster_id=cid, name=name, method=method, members=members)


def _pfm_from_json(matrix_data_json, matrix_type):
    """Parse MATRIX_DATA (JSON list of dicts with A/C/G/T) and return (PFM ndarray, kind)."""
    obj = json.loads(matrix_data_json)
    if not isinstance(obj, list) or not obj or not isinstance(obj[0], dict):
        raise ValueError("DL MATRIX_DATA must be a JSON list of dicts with A,C,G,T keys.")
    arr = np.array([[float(p["A"]), float(p["C"]), float(p["G"]), float(p["T"])]
                    for p in obj], dtype=float)

    mtype = (matrix_type or "").upper()
    if mtype == "PFM":
        pfm = np.clip(arr, 0.0, None)
        return pfm, "pfm"

    # CWM → pseudo-PFM: magnitude -> per-position normalize -> scale to ~100 + pseudo
    cwm = np.abs(arr)
    denom = cwm.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    probs = cwm / denom
    scale, pseudo = 100.0, 0.5
    pfm = probs * scale + pseudo
    return pfm, "cwm-as-pfm"
