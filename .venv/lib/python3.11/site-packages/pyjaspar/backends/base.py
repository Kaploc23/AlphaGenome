
class BackendBase:
    """Minimal backend base.

    Stores a DB engine-like object (SQLAlchemy engine) and options.
    """
    def __init__(self, engine=None, **opts):
        self.engine = engine
        self.opts = opts
