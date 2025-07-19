class ScarcityEngine:
    def __init__(self, total_supply=21000000):
        self.total_supply = total_supply
        self.minted = 0
        self.burned = 0
    def mint_span(self, span):
        if self.minted >= self.total_supply:
            return False
        difficulty = self._calculate_difficulty()
        if span.energy < difficulty:
            return False
        self.minted += 1
        self._adjust_difficulty()
        return True
    def burn_span(self, span_id):
        self.burned += 1
        self._adjust_difficulty()
    def _calculate_difficulty(self):
        circulating = self.minted - self.burned
        return max(10.0, circulating * 0.0001)
    def _adjust_difficulty(self):
        if self.minted % 1050000 == 0:
            self.total_supply //= 2
    def decay_all(self, ledger):
        for span in ledger.values():
            span.decay()