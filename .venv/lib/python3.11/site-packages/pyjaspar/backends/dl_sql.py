
"""DL backend for PyJASPAR (SQLite-friendly).

- Reads DL_* tables from the same JASPAR2026.sqlite used by CORE.

- Produces public Matrix/Cluster objects via adapters.
"""
from .base import BackendBase
# Reuse CORE's engine resolution if no engine provided
from .sqlite import SQLiteBackend

from ..models.adapters_dl import dl_row_to_matrix, dl_cluster_rows_to_cluster
from ..utils import split_dl_acc


class DLSQLBackend(BackendBase):
    """Deep Learning (DL) collection backend operating on DL_* tables."""

    collection_name = "DL"

    def __init__(self, engine=None, **opts):
        # Fall back to the same SQLite engine as CORE
        if engine is None:
            core = SQLiteBackend(engine=None, **opts)
            engine = core.engine
        super().__init__(engine=engine, **opts)

    # ------------- Public API -------------
    def fetch_motifs(self, **filters):
        sql, params = self._build_matrix_query(**filters)
        rows = self._execute(sql, params)
        return [dl_row_to_matrix(r) for r in rows]

    def fetch_matrix(self, matrix_id):
        base_id, version = split_dl_acc(matrix_id)
        sql = """

        WITH primary_model AS (

          SELECT * FROM DL_MODEL WHERE BASE_ID = ? AND VERSION = ?

        )

        SELECT

          mm.ID               AS matrix_row_id,
          mm.MOTIF_ID         AS motif_id,
          mm.MATRIX_TYPE      AS matrix_type,
          mm.MATRIX_DATA      AS matrix_data,
          mo.ID               AS dl_motif_id,
          mo.NAME             AS motif_name,
          mo.SOURCE           AS motif_source,
          mo.SOURCE_ID        AS motif_source_id,
          mo.SOURCE_VERSION   AS motif_source_version,
          mo.NUM_SEQLETS      AS num_seqlets,
          pm.BASE_ID          AS base_id,
          pm.VERSION          AS version,
          pm.TF_NAME          AS tf_name,
          pm.CELL_LINE        AS cell_line,
          pm.DATA_TYPE        AS data_type,
          pm.MODEL_NAME       AS model_name,
          pm.SOURCE           AS model_source,
          pm.SOURCE_ID        AS model_source_id,
          pm.SOURCE_URL       AS model_source_url,
          tx.TAX_ID           AS tax_id,
          tx.SPECIES          AS species,
          tx.TAX_GROUP        AS tax_group,
          tfa.CLASS           AS tf_class,
          tfa.FAMILY          AS tf_family,
          tfa.UNIPROT_ID      AS uniprot_id

        FROM DL_MOTIF mo

        JOIN DL_MATRIX mm ON mm.MOTIF_ID = mo.ID

        LEFT JOIN primary_model pm ON pm.PRIMARY_MOTIF_ID = mo.ID

        LEFT JOIN DL_TAXONOMY tx ON tx.TAX_ID = pm.TAX_ID

        LEFT JOIN DL_TF_ANNOTATION tfa ON tfa.TF_NAME = pm.TF_NAME

        WHERE pm.BASE_ID = ? AND pm.VERSION = ?

        ORDER BY CASE WHEN mm.MATRIX_TYPE='PFM' THEN 0 ELSE 1 END

        LIMIT 1

        """
        params = [base_id, version, base_id, version]
        row = self._execute_one(sql, params)
        return dl_row_to_matrix(row) if row else None

    def fetch_clusters(self, **filters):
        sql, params = self._build_cluster_query(**filters)
        rows = self._execute(sql, params)
        groups = {}
        for r in rows:
            cid = r["cluster_id"]
            groups.setdefault(cid, {"cluster": r, "members": []})
            groups[cid]["members"].append(r)
        return {cid: dl_cluster_rows_to_cluster(g) for cid, g in groups.items()}

    # ------------- Query builders -------------
    def _build_matrix_query(self, **f):
        base = """

        SELECT

          mm.ID               AS matrix_row_id,

          mm.MOTIF_ID         AS motif_id,

          mm.MATRIX_TYPE      AS matrix_type,

          mm.MATRIX_DATA      AS matrix_data,

          mo.ID               AS dl_motif_id,

          mo.NAME             AS motif_name,

          mo.SOURCE           AS motif_source,

          mo.SOURCE_ID        AS motif_source_id,

          mo.SOURCE_VERSION   AS motif_source_version,

          mo.NUM_SEQLETS      AS num_seqlets,

          md.BASE_ID          AS base_id,

          md.VERSION          AS version,

          md.TF_NAME          AS tf_name,

          md.CELL_LINE        AS cell_line,

          md.DATA_TYPE        AS data_type,

          md.MODEL_NAME       AS model_name,

          md.SOURCE           AS model_source,

          md.SOURCE_ID        AS model_source_id,

          md.SOURCE_URL       AS model_source_url,

          tx.TAX_ID           AS tax_id,

          tx.SPECIES          AS species,

          tx.TAX_GROUP        AS tax_group,

          tfa.CLASS           AS tf_class,

          tfa.FAMILY          AS tf_family,

          tfa.UNIPROT_ID      AS uniprot_id

        FROM DL_MATRIX mm

        JOIN DL_MOTIF  mo ON mo.ID = mm.MOTIF_ID

        LEFT JOIN DL_MODEL md ON md.PRIMARY_MOTIF_ID = mo.ID

        LEFT JOIN DL_TAXONOMY tx ON tx.TAX_ID = md.TAX_ID

        LEFT JOIN DL_TF_ANNOTATION tfa ON tfa.TF_NAME = md.TF_NAME

        """
        where, params = [], []
        self._in_clause(where, params, "tx.TAX_ID",    f.get("species"))
        self._in_clause(where, params, "tx.TAX_GROUP", f.get("tax_group"))
        self._in_clause(where, params, "md.TF_NAME",   f.get("tf_name"))
        self._in_clause(where, params, "tfa.CLASS",    f.get("tf_class"))
        self._in_clause(where, params, "tfa.FAMILY",   f.get("tf_family"))
        self._in_clause(where, params, "md.CELL_LINE", f.get("cell_line"))
        self._in_clause(where, params, "md.DATA_TYPE", f.get("data_type"))
        self._in_clause(where, params, "mm.MATRIX_TYPE", f.get("matrix_type"))  # ['PFM','CWM']

        sql = base
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY md.BASE_ID, md.VERSION, CASE WHEN mm.MATRIX_TYPE='PFM' THEN 0 ELSE 1 END"
        return sql, params

    def _build_cluster_query(self, **f):
        base = """

        SELECT

          pr.ID               AS cluster_id,

          pr.BASE_ID          AS base_id,

          pr.VERSION          AS version,

          pr.PRIMARY_MOTIF_ID AS primary_motif_id,

          pr.TF_NAME          AS tf_name,

          pr.TAX_GROUP        AS cluster_tax_group,



          mm.ID               AS matrix_row_id,

          mm.MOTIF_ID         AS motif_id,

          mm.MATRIX_TYPE      AS matrix_type,

          mm.MATRIX_DATA      AS matrix_data,



          mo.ID               AS dl_motif_id,

          mo.NAME             AS motif_name,

          mo.SOURCE           AS motif_source,

          mo.SOURCE_ID        AS motif_source_id,

          mo.SOURCE_VERSION   AS motif_source_version,

          mo.NUM_SEQLETS      AS num_seqlets,



          md.BASE_ID          AS member_base_id,

          md.VERSION          AS member_version,

          md.TF_NAME          AS member_tf_name,

          md.CELL_LINE        AS cell_line,

          md.DATA_TYPE        AS data_type,

          md.MODEL_NAME       AS model_name,

          md.SOURCE           AS model_source,

          md.SOURCE_ID        AS model_source_id,

          md.SOURCE_URL       AS model_source_url,



          tx.TAX_ID           AS tax_id,

          tx.SPECIES          AS species,

          tx.TAX_GROUP        AS tax_group,

          tfa.CLASS           AS tf_class,

          tfa.FAMILY          AS tf_family,

          tfa.UNIPROT_ID      AS uniprot_id

        FROM DL_PRIMARY_CLUSTER pr

        LEFT JOIN DL_MOTIF mo ON mo.ID = pr.PRIMARY_MOTIF_ID

        LEFT JOIN DL_MATRIX mm ON mm.MOTIF_ID = mo.ID

        LEFT JOIN DL_MODEL md ON md.PRIMARY_MOTIF_ID = mo.ID

        LEFT JOIN DL_TAXONOMY tx ON tx.TAX_ID = md.TAX_ID

        LEFT JOIN DL_TF_ANNOTATION tfa ON tfa.TF_NAME = md.TF_NAME

        """
        where, params = [], []
        self._in_clause(where, params, "pr.TAX_GROUP", f.get("tax_group"))
        self._in_clause(where, params, "pr.TF_NAME",   f.get("tf_name"))
        sql = base
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY pr.BASE_ID, pr.VERSION, CASE WHEN mm.MATRIX_TYPE='PFM' THEN 0 ELSE 1 END"
        return sql, params

    # ------------- Exec helpers & IN clause -------------
    def _execute(self, sql, params=None):
        with self.engine.connect() as con:
            cur = con.exec_driver_sql(sql, tuple(params or ()))
            cols = cur.keys()
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _execute_one(self, sql, params=None):
        with self.engine.connect() as con:
            row = con.exec_driver_sql(sql, tuple(params or ())).fetchone()
            if not row:
                return None
            cols = row.keys()
            return dict(zip(cols, row))

    def _in_clause(self, where, params, col, values):
        if not values:
            return
        placeholders = ",".join(["?"] * len(values))
        where.append(f"{col} IN ({placeholders})")
        params.extend(values)
