"""
Span Algebra implementation for LogLine/DiamondSpan
"""
from typing import Dict, List, Any, Optional
import hashlib
import json
import time
from datetime import datetime

class SpanAlgebra:
    @staticmethod
    def compose(span1: Dict[str, Any], span2: Dict[str, Any]) -> Dict[str, Any]:
        """Compose two spans: S₁ ∘ S₂"""
        # Ensure spans are compatible for composition
        if span1.get('object') != span2.get('verb'):
            raise ValueError("Composition requires span1.object == span2.verb")
            
        result_span = {
            'kind': 'composite',
            'verb': span1['verb'],
            'who': f"{span1['who']}∘{span2['who']}",
            'object': span2['object'],
            'what': f"{span1['what']}_then_{span2['what']}",
            'why': f"composition of {span1['id']} and {span2['id']}",
            'parent_ids': [span1['id'], span2['id']],
            'payload': {
                'components': [span1['id'], span2['id']],
                'operation': 'compose'
            }
        }
        
        return result_span
        
    @staticmethod
    def superpose(span1: Dict[str, Any], span2: Dict[str, Any]) -> Dict[str, Any]:
        """Superposition of two spans: S₁ ⊕ S₂"""
        # Ensure spans have compatible objects for superposition
        if span1.get('object') != span2.get('object'):
            raise ValueError("Superposition requires span1.object == span2.object")
            
        result_span = {
            'kind': 'superposition',
            'verb': f"{span1['verb']}+{span2['verb']}",
            'who': f"{span1['who']}⊕{span2['who']}",
            'object': span1['object'],
            'what': f"{span1['what']}_and_{span2['what']}",
            'why': f"superposition of {span1['id']} and {span2['id']}",
            'parent_ids': [span1['id'], span2['id']],
            'payload': {
                'components': [span1['id'], span2['id']],
                'operation': 'superpose'
            }
        }
        
        return result_span
        
    @staticmethod
    def negate(span: Dict[str, Any]) -> Dict[str, Any]:
        """Negation of a span: ¬S"""
        result_span = {
            'kind': 'negation',
            'verb': f"NOT_{span['verb']}",
            'who': f"anti_{span['who']}",
            'object': span['object'],
            'what': f"undo_{span['what']}",
            'why': f"negation of {span['id']}",
            'parent_ids': [span['id']],
            'payload': {
                'target': span['id'],
                'operation': 'negate'
            }
        }
        
        return result_span
        
    @staticmethod
    def collapse(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collapse multiple spans into a single span"""
        if not spans:
            raise ValueError("Cannot collapse empty span list")
            
        # Extract common object if possible
        objects = set(span.get('object') for span in spans)
        if len(objects) == 1:
            common_object = list(objects)[0]
        else:
            common_object = "multi_object"
            
        verbs = [span.get('verb') for span in spans]
        actors = [span.get('who') for span in spans]
        actions = [span.get('what') for span in spans]
        
        result_span = {
            'kind': 'collapse',
            'verb': f"COLLAPSE({'_'.join(verbs)})",
            'who': f"collective({'_'.join(actors)})",
            'object': common_object,
            'what': f"collapse({'_'.join(actions)})",
            'why': f"collapse of {len(spans)} spans",
            'parent_ids': [span['id'] for span in spans],
            'payload': {
                'components': [span['id'] for span in spans],
                'operation': 'collapse'
            }
        }
        
        return result_span
        
    @staticmethod
    def evaluate_energy(span: Dict[str, Any]) -> float:
        """Evaluate the energy of a span"""
        # Base energy calculation
        base_energy = 10.0
        
        # Adjust based on span complexity
        payload_complexity = len(json.dumps(span.get('payload', {}))) / 100
        parent_count = len(span.get('parent_ids', []))
        
        # Formula: E = base * (1 + payload_factor) * (1 + parent_factor)
        energy = base_energy * (1 + payload_complexity * 0.1) * (1 + parent_count * 0.05)
        
        # Apply adjustments for special span types
        kind = span.get('kind', '')
        if kind == 'collapse':
            energy *= 1.5  # Collapse operations are more energy-intensive
        elif kind == 'negation':
            energy *= 0.8  # Negations are simpler
            
        # Cap energy at reasonable values
        return min(100.0, energy)
        
    @staticmethod
    def evaluate_tension(span: Dict[str, Any]) -> float:
        """Evaluate the tension of a span"""
        # Base tension
        base_tension = 1.0
        
        # Adjust based on verb
        verb = span.get('verb', '').lower()
        high_tension_verbs = ['delete', 'destroy', 'remove', 'kill', 'attack', 'hack']
        for htv in high_tension_verbs:
            if htv in verb:
                base_tension *= 3.0
                break
                
        # Adjust based on payload content
        payload_str = json.dumps(span.get('payload', {})).lower()
        risk_terms = ['weapon', 'harmful', 'danger', 'attack', 'unauthorized']
        for term in risk_terms:
            if term in payload_str:
                base_tension *= 2.0
                
        # Adjust based on span type
        kind = span.get('kind', '')
        if kind in ['emergency', 'critical', 'override']:
            base_tension *= 1.5
            
        # Apply decay based on time (older spans have less tension)
        created_at = span.get('created_at', time.time())
        time_factor = max(0.5, min(1.0, (time.time() - created_at) / (3600 * 24)))
        
        # Final tension calculation
        tension = base_tension * time_factor
        
        return min(20.0, tension)  # Cap at 20.0 (above the 17.3 threshold)