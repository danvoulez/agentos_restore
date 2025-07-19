def generate_causal_hash(content):
    import hashlib
    return hashlib.sha256(content.encode()).hexdigest()

def calculate_authority(authority):
    # Placeholder: return some score
    return 1.0

def get_ancestral_spans():
    return ["span1", "span2"]

class DiamondSpan:
    def __init__(self, id, verb, actor, object, payload):
        self.id = id
        self.verb = verb
        self.actor = actor
        self.object = object
        self.payload = payload

class DiamondSpanFactory:
    def generate(self, content, authority):
        return DiamondSpan(
            id=generate_causal_hash(content),
            verb="CONHECIMENTO",
            actor=authority,
            object="capital_cognitivo",
            payload={
                "content": content,
                "authority_score": calculate_authority(authority),
                "applicability": ["treino", "fine_tune"],
                "provenance_chain": get_ancestral_spans()
            }
        )