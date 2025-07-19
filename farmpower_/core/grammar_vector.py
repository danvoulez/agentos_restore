"""
Grammar Vector implementation for LogLine/DiamondSpan
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import hashlib

class Rule:
    def __init__(self, subject: str, predicate: str, object_: str, weight: float = 1.0):
        self.subject = subject
        self.predicate = predicate
        self.object = object_
        self.weight = weight
        
    def __repr__(self):
        return f"Rule({self.subject}, {self.predicate}, {self.object}, {self.weight})"
        
    def to_vector(self, dimension: int = 128) -> np.ndarray:
        """Convert rule to vector representation"""
        # Hash-based encoding
        content = f"{self.subject}:{self.predicate}:{self.object}"
        hash_bytes = hashlib.sha256(content.encode()).digest()
        
        # Convert to vector of specified dimension
        indices = [hash_bytes[i] % dimension for i in range(min(16, dimension))]
        vector = np.zeros(dimension)
        for idx in indices:
            vector[idx] = 1.0
            
        # Scale by weight
        vector *= self.weight
        return vector
        
class GrammarVector:
    def __init__(self, verb: str = None, object_: str = None):
        self.rules: List[Rule] = []
        self.dimension = 256  # Default dimension
        
        if verb and object_:
            self.add_rule("span", verb, object_)
            
    def add_rule(self, subject: str, predicate: str, object_: str, weight: float = 1.0):
        """Add a grammatical rule"""
        rule = Rule(subject, predicate, object_, weight)
        self.rules.append(rule)
        
    def to_vector(self) -> np.ndarray:
        """Convert all rules to a combined vector representation"""
        if not self.rules:
            return np.zeros(self.dimension)
            
        # Combine all rule vectors
        vectors = [rule.to_vector(self.dimension) for rule in self.rules]
        return np.sum(vectors, axis=0)
        
    def cosine_similarity(self, other: 'GrammarVector') -> float:
        """Calculate cosine similarity with another grammar vector"""
        vec1 = self.to_vector()
        vec2 = other.to_vector()
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
        
    def generate_language(self, complexity: int = 3) -> List[str]:
        """Generate natural language from grammar rules"""
        sentences = []
        
        for _ in range(complexity):
            rules = np.random.choice(self.rules, min(3, len(self.rules)), replace=False)
            sentence = " ".join([f"{r.subject} {r.predicate} {r.object}" for r in rules])
            sentences.append(sentence.capitalize() + ".")
            
        return sentences
        
    @staticmethod
    def from_natural_language(text: str) -> 'GrammarVector':
        """Extract grammar vector from natural language text"""
        # This is a simplified implementation
        gv = GrammarVector()
        
        # Split into sentences and process
        sentences = text.split('.')
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) >= 3:
                # Simple SVO extraction
                subject = words[0]
                predicate = words[1] if len(words) > 1 else "is"
                object_ = words[2] if len(words) > 2 else "unknown"
                
                gv.add_rule(subject, predicate, object_)
                
        return gv