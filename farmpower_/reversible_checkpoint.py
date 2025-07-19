import hashlib

def sha256(x):
    return hashlib.sha256(x).hexdigest()

class SpanDiamante:
    pass

class TensorOpSpan:
    def __init__(self, op, params):
        self.op = op
        self.params = params

class ReversibleCheckpoint(SpanDiamante):
    def __init__(self, epoch_state, prev_epoch=None):
        super().__init__()
        self.verb = "CHECKPOINT"
        self.object = f"epoch_{epoch_state.id}"
        self.payload = {
            "weights_hash": sha256(epoch_state.weights),
            "tensor_diffs": self.compute_reversible_delta(prev_epoch)
        }

    def compute_reversible_delta(self, prev_epoch):
        if prev_epoch is None:
            return 0
        # Placeholder for diff computation
        return epoch_state.weights - prev_epoch.weights

    def compensate(self):
        """Rollback para estado anterior"""
        return TensorOpSpan(
            op="APPLY_DELTA",
            params={"delta": -self.payload["tensor_diffs"]}
        )