
"""Convenience bootstrap for using the DL backend without touching jaspardb.py."""
from .backends.sqlite import SQLiteBackend
from .backends.dl_sql import DLSQLBackend

class jaspardb_dl:
    """Drop-in facade mirroring jaspardb but defaulting to the DL backend."""
    def __init__(self, engine=None, **opts):
        # Reuse CORE engine if not provided
        if engine is None:
            core = SQLiteBackend(engine=None, **opts)
            engine = core.engine
        self.backend = DLSQLBackend(engine=engine, **opts)

    def fetch_motifs(self, **filters):
        return self.backend.fetch_motifs(**filters)

    def fetch_matrix(self, matrix_id):
        return self.backend.fetch_matrix(matrix_id)

    def fetch_clusters(self, **filters):
        return self.backend.fetch_clusters(**filters)
