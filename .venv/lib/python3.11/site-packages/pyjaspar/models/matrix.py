
import numpy as np

class Matrix:
    """Lightweight stand-in for PyJASPAR's public Matrix type.
    Stores a PFM-like numpy array and metadata. Only minimal methods used here.
    Replace with your real class if available.
    """
    def __init__(self, matrix_id, name, collection, pfm, metadata=None):
        self.matrix_id = matrix_id
        self.name = name
        self.collection = collection
        self._pfm = np.array(pfm, dtype=float)
        self.metadata = metadata or {}

    def pfm(self):
        return self._pfm

    def to_pwm(self, bg=None, pseudocount=0.8):
        """Convert PFM to PWM (log odds). bg: dict with A,C,G,T probs."""
        if bg is None:
            bg = {"A":0.25, "C":0.25, "G":0.25, "T":0.25}
        bg_vec = np.array([bg["A"], bg["C"], bg["G"], bg["T"]], dtype=float)
        counts = self._pfm + pseudocount
        probs = counts / counts.sum(axis=1, keepdims=True)
        pwm = np.log((probs + 1e-12) / bg_vec)
        return pwm

class Cluster:
    """Minimal Cluster container with members (list of Matrix)."""
    def __init__(self, cluster_id, name=None, method=None, members=None):
        self.cluster_id = cluster_id
        self.name = name or cluster_id
        self.method = method or "DL"
        self.members = members or []
