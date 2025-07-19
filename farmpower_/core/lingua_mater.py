"""
Lingua Mater implementation for LogLineOS/DiamondSpan
The native ontology language for LogLineOS
"""
import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass

@dataclass
class LinguaTerm:
    """A term in Lingua Mater"""
    symbol: str
    semantic_type: str
    vector: List[float]
    relations: Dict[str, List[str]]

@dataclass
class LinguaRule:
    """A grammar rule in Lingua Mater"""
    left: str
    right: List[str]
    weight: float
    constraints: Dict[str, Any]

class LinguaMater:
    """
    Lingua Mater implementation - the ontological foundation of LogLineOS
    """
    
    def __init__(self):
        self.terms: Dict[str, LinguaTerm] = {}
        self.rules: List[LinguaRule] = []
        self.ontology_hash = ""
        self.version = "1.0.0"
    
    def add_term(self, symbol: str, semantic_type: str, vector: List[float] = None, 
                 relations: Dict[str, List[str]] = None):
        """Add a term to the ontology"""
        if not vector:
            # Generate a random vector if none provided
            import random
            vector = [random.uniform(-1, 1) for _ in range(128)]
        
        term = LinguaTerm(
            symbol=symbol,
            semantic_type=semantic_type,
            vector=vector,
            relations=relations or {}
        )
        
        self.terms[symbol] = term
        self._update_hash()
        
        return term
    
    def add_rule(self, left: str, right: List[str], weight: float = 1.0, 
                 constraints: Dict[str, Any] = None):
        """Add a grammar rule to the ontology"""
        rule = LinguaRule(
            left=left,
            right=right,
            weight=weight,
            constraints=constraints or {}
        )
        
        self.rules.append(rule)
        self._update_hash()
        
        return rule
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse text using Lingua Mater grammar"""
        # Split into tokens
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        # Apply grammar rules to construct parse tree
        parse_tree = self._construct_parse_tree(tokens)
        
        return parse_tree
    
    def generate(self, semantic_structure: Dict[str, Any]) -> str:
        """Generate text from semantic structure using Lingua Mater"""
        # This is a simplified implementation
        if "verb" in semantic_structure and "object" in semantic_structure:
            verb = semantic_structure["verb"]
            obj = semantic_structure["object"]
            
            # Simple template-based generation
            return f"The system will {verb.lower()} the {obj.lower()}"
        
        return json.dumps(semantic_structure)
    
    def to_span(self, text: str) -> Dict[str, Any]:
        """Convert text to a Diamond Span using Lingua Mater parsing"""
        # Parse the text
        parsed = self.parse(text)
        
        # Extract key components
        kind = parsed.get("kind", "text")
        verb = parsed.get("verb", "STATE")
        actor = parsed.get("actor", "system")
        object_ = parsed.get("object", "content")
        
        # Create span structure
        span = {
            "kind": kind,
            "verb": verb,
            "actor": actor,
            "object": object_,
            "payload": {
                "text": text,
                "parsed_structure": parsed,
                "grammar_complexity": self._calculate_complexity(parsed)
            },
            "metadata": {
                "lingua_mater_version": self.version,
                "ontology_hash": self.ontology_hash
            }
        }
        
        return span
    
    def _construct_parse_tree(self, tokens: List[str]) -> Dict[str, Any]:
        """Construct a parse tree from tokens using grammar rules"""
        # This is a simplified implementation
        # A real implementation would use a proper parser
        
        # Try to identify key components
        tree = {"tokens": tokens}
        
        # Try to find verb
        verbs = [t for t in tokens if self._is_of_type(t, "verb")]
        if verbs:
            tree["verb"] = verbs[0]
        
        # Try to find object
        nouns = [t for t in tokens if self._is_of_type(t, "noun")]
        if nouns:
            tree["object"] = nouns[0]
        
        # Try to identify actor
        actors = [t for t in tokens if self._is_of_type(t, "actor")]
        if actors:
            tree["actor"] = actors[0]
        
        return tree
    
    def _is_of_type(self, token: str, semantic_type: str) -> bool:
        """Check if a token is of a particular semantic type"""
        # Check in known terms
        if token.lower() in self.terms:
            return self.terms[token.lower()].semantic_type == semantic_type
        
        # Simple heuristics as fallback
        if semantic_type == "verb":
            verb_endings = ["e", "ate", "ize", "fy", "en", "ing"]
            return any(token.lower().endswith(end) for end in verb_endings)
        
        if semantic_type == "noun":
            noun_endings = ["tion", "ness", "ity", "ment", "er", "or"]
            return any(token.lower().endswith(end) for end in noun_endings)
        
        return False
    
    def _calculate_complexity(self, parse_tree: Dict[str, Any]) -> float:
        """Calculate grammatical complexity score of a parse tree"""
        # Simple complexity metric
        complexity = 1.0
        
        # More components = more complex
        complexity += len(parse_tree.keys()) * 0.5
        
        # More tokens = more complex
        complexity += len(parse_tree.get("tokens", [])) * 0.1
        
        return min(10.0, complexity)  # Cap at 10
    
    def _update_hash(self):
        """Update the ontology hash"""
        # Create a deterministic representation of the ontology
        terms_str = json.dumps(sorted([t.symbol for t in self.terms.values()]))
        rules_str = json.dumps([(r.left, str(r.right), r.weight) for r in self.rules])
        
        # Hash it
        combined = f"{terms_str}|{rules_str}|{self.version}"
        self.ontology_hash = hashlib.sha256(combined.encode()).hexdigest()
    
    def initialize_core_ontology(self):
        """Initialize the core ontology for LogLineOS"""
        # Add basic semantic types
        basic_types = [
            ("span", "entity"),
            ("verb", "type"),
            ("noun", "type"),
            ("actor", "type"),
            ("object", "type"),
            ("CREATE", "verb"),
            ("MODIFY", "verb"),
            ("READ", "verb"),
            ("DELETE", "verb"),
            ("EXECUTE", "verb"),
            ("ESTABLISH", "verb"),
            ("system", "actor"),
            ("user", "actor"),
            ("administrator", "actor"),
            ("danvoulez", "actor")
        ]
        
        for symbol, semantic_type in basic_types:
            self.add_term(symbol, semantic_type)
        
        # Add basic grammar rules
        self.add_rule("S", ["NP", "VP"], 1.0)
        self.add_rule("NP", ["Det", "N"], 0.8)
        self.add_rule("NP", ["N"], 0.5)
        self.add_rule("VP", ["V", "NP"], 1.0)
        self.add_rule("VP", ["V"], 0.5)
        
        self._update_hash()
        return self