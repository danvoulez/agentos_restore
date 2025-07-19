"""
Causal DAG implementation for LogLine/DiamondSpan
"""
from typing import Dict, List, Set, Any, Optional
import hashlib
import time
import networkx as nx
import matplotlib.pyplot as plt

class CausalDAG:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, List[str]] = {}  # parent -> children
        self.reverse_edges: Dict[str, List[str]] = {}  # child -> parents
        
    def add_node(self, node_id: str, metadata: Dict[str, Any] = None) -> None:
        """Add a node to the causal DAG"""
        if node_id in self.nodes:
            return
            
        self.nodes[node_id] = metadata or {}
        self.edges[node_id] = []
        self.reverse_edges[node_id] = []
        
    def add_edge(self, parent_id: str, child_id: str) -> None:
        """Add a causal edge from parent to child"""
        # Ensure nodes exist
        if parent_id not in self.nodes:
            self.add_node(parent_id)
        if child_id not in self.nodes:
            self.add_node(child_id)
            
        # Add edge
        if child_id not in self.edges[parent_id]:
            self.edges[parent_id].append(child_id)
        if parent_id not in self.reverse_edges[child_id]:
            self.reverse_edges[child_id].append(parent_id)
            
    def get_ancestors(self, node_id: str) -> Set[str]:
        """Get all ancestors of a node"""
        if node_id not in self.nodes:
            return set()
            
        ancestors = set()
        queue = self.reverse_edges.get(node_id, []).copy()
        
        while queue:
            parent = queue.pop()
            ancestors.add(parent)
            queue.extend([p for p in self.reverse_edges.get(parent, []) if p not in ancestors])
            
        return ancestors
        
    def get_descendants(self, node_id: str) -> Set[str]:
        """Get all descendants of a node"""
        if node_id not in self.nodes:
            return set()
            
        descendants = set()
        queue = self.edges.get(node_id, []).copy()
        
        while queue:
            child = queue.pop()
            descendants.add(child)
            queue.extend([c for c in self.edges.get(child, []) if c not in descendants])
            
        return descendants
        
    def is_ancestor(self, potential_ancestor: str, node_id: str) -> bool:
        """Check if one node is an ancestor of another"""
        return potential_ancestor in self.get_ancestors(node_id)
        
    def get_causal_cone(self, node_id: str) -> Dict[str, Any]:
        """Get the causal cone (ancestors and their metadata) of a node"""
        ancestors = self.get_ancestors(node_id)
        return {
            ancestor: self.nodes[ancestor]
            for ancestor in ancestors
        }
        
    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the DAG (which shouldn't exist in a valid causal DAG)"""
        # Convert to networkx graph for cycle detection
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node)
        for parent, children in self.edges.items():
            for child in children:
                G.add_edge(parent, child)
                
        return list(nx.simple_cycles(G))
        
    def visualize(self, output_path: Optional[str] = None):
        """Visualize the causal DAG"""
        G = nx.DiGraph()
        
        # Add nodes
        for node, metadata in self.nodes.items():
            label = metadata.get('label', node[:8])
            G.add_node(node, label=label)
            
        # Add edges
        for parent, children in self.edges.items():
            for child in children:
                G.add_edge(parent, child)
                
        # Create layout
        pos = nx.spring_layout(G)
        
        # Draw graph
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes})
        
        plt.title("Causal DAG Visualization")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()