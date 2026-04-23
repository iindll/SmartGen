"""
smartgen_gss.py - Complete GSS Module for SmartGen
Add this file to your SmartGen folder - works with existing code
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
from sklearn.cluster import KMeans


class SmartGenGSS:
    """
    Complete GSS with:
    - Behavior Graph Construction
    - Behavior Matrix Extraction  
    - Semantic Space Mapping
    - JSON Hints Generation
    - Sequence Compression
    """
    
    def __init__(self, actions_dict: Dict[str, int]):
        self.actions_dict = actions_dict
        self.id_to_action = {v: k for k, v in actions_dict.items()}
        self.graph = {}  # behavior graph
        self.transition_matrix = None
        self.semantic_clusters = {}
        self.json_hints = {}
        
        # Semantic categories for mapping
        self.semantic_categories = {
            'comfort': ['temperature', 'heat', 'cool', 'fan', 'humidity', 'air', 'thermostat', 'ac'],
            'lighting': ['light', 'bulb', 'dim', 'bright', 'color', 'lamp', 'blind', 'switch'],
            'security': ['lock', 'camera', 'alarm', 'motion', 'sensor', 'door', 'secure'],
            'entertainment': ['tv', 'audio', 'speaker', 'music', 'video', 'play', 'stream'],
            'appliance': ['oven', 'microwave', 'fridge', 'dishwasher', 'washer', 'dryer', 'clean'],
            'automation': ['schedule', 'timer', 'auto', 'routine', 'scene', 'mode'],
            'energy': ['power', 'meter', 'solar', 'battery', 'plug', 'outlet', 'energy']
        }
    
    def build_graph(self, sequences: List[List[int]]) -> Dict:
        """Build behavior graph from action sequences"""
        print("\n[GSS] Building Behavior Graph...")
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                frm = seq[i]
                to = seq[i + 1]
                
                if frm not in self.graph:
                    self.graph[frm] = {}
                if to not in self.graph[frm]:
                    self.graph[frm][to] = 0
                self.graph[frm][to] += 1
        
        nodes = len(self.graph)
        edges = sum(len(v) for v in self.graph.values())
        print(f"  Graph: {nodes} nodes, {edges} edges")
        return self.graph
    
    def extract_matrix(self, num_actions: int) -> np.ndarray:
        """Extract transition probability matrix"""
        print("[GSS] Extracting Behavior Matrix...")
        
        matrix = np.zeros((num_actions, num_actions))
        
        for frm, transitions in self.graph.items():
            total = sum(transitions.values())
            if total > 0:
                for to, count in transitions.items():
                    if frm < num_actions and to < num_actions:
                        matrix[frm, to] = count / total
        
        self.transition_matrix = matrix
        print(f"  Matrix shape: {matrix.shape}")
        return matrix
    
    def semantic_mapping(self) -> Dict[int, int]:
        """Map actions to semantic clusters"""
        print("[GSS] Performing Semantic Space Mapping...")
        
        # Extract features for each action
        action_ids = list(self.actions_dict.values())
        features = []
        valid_ids = []
        
        for aid in action_ids:
            action_str = self.id_to_action.get(aid, "")
            feat = self._extract_semantic_features(action_str)
            if np.sum(feat) > 0:
                features.append(feat)
                valid_ids.append(aid)
        
        if len(features) >= 2:
            n_clusters = min(8, len(features))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            self.semantic_clusters = {aid: int(clusters[i]) for i, aid in enumerate(valid_ids)}
        else:
            self.semantic_clusters = {aid: 0 for aid in action_ids}
        
        # Print summary
        cluster_counts = Counter(self.semantic_clusters.values())
        print(f"  Created {len(cluster_counts)} semantic clusters")
        for cid, count in sorted(cluster_counts.items())[:5]:
            print(f"    Cluster {cid}: {count} actions")
        
        return self.semantic_clusters
    
    def _extract_semantic_features(self, action_str: str) -> np.ndarray:
        """Extract semantic feature vector for an action"""
        features = np.zeros(len(self.semantic_categories))
        action_lower = action_str.lower()
        
        for i, (category, keywords) in enumerate(self.semantic_categories.items()):
            for keyword in keywords:
                if keyword in action_lower:
                    features[i] += 1
        
        if np.sum(features) > 0:
            features = features / np.sum(features)
        return features
    
    def generate_hints(self) -> Dict:
        """Generate JSON hints for LLM"""
        print("[GSS] Generating JSON Hints...")
        
        # Build cluster info
        cluster_info = {}
        for aid, cid in self.semantic_clusters.items():
            if cid not in cluster_info:
                cluster_info[cid] = {"actions": [], "size": 0}
            action_str = self.id_to_action.get(aid, f"id_{aid}")
            cluster_info[cid]["actions"].append(action_str)
            cluster_info[cid]["size"] += 1
        
        # Build top transitions
        top_transitions = []
        for frm in list(self.graph.keys())[:30]:
            if frm in self.graph and self.graph[frm]:
                sorted_trans = sorted(self.graph[frm].items(), key=lambda x: x[1], reverse=True)[:3]
                total = sum(self.graph[frm].values())
                for to, cnt in sorted_trans:
                    top_transitions.append({
                        "from": self.id_to_action.get(frm, f"id_{frm}"),
                        "to": self.id_to_action.get(to, f"id_{to}"),
                        "count": cnt,
                        "probability": cnt / total if total > 0 else 0
                    })
        
        self.json_hints = {
            "graph_stats": {
                "nodes": len(self.graph),
                "edges": sum(len(v) for v in self.graph.values())
            },
            "semantic_clusters": {
                str(cid): {"size": info["size"], "examples": info["actions"][:3]}
                for cid, info in cluster_info.items()
            },
            "top_transitions": top_transitions[:15],
            "generation_rules": [
                "Prefer transitions within same semantic cluster",
                "Morning: lighting first, then comfort actions",
                "Night: security actions before leaving",
                "Maintain logical device sequences"
            ]
        }
        
        print(f"  Generated {len(self.json_hints)} hint sections")
        return self.json_hints
    
    def compress(self, sequences: List[List[int]], ratio: float = 0.7) -> List[List[int]]:
        """Compress sequences by keeping high-probability transitions"""
        print(f"[GSS] Compressing sequences (ratio={ratio})...")
        
        compressed = []
        for seq in sequences:
            if len(seq) <= 2:
                compressed.append(seq)
                continue
            
            filtered = [seq[0]]
            for i in range(1, len(seq)):
                frm, to = seq[i-1], seq[i]
                
                prob = 0
                if frm in self.graph and to in self.graph.get(frm, {}):
                    total = sum(self.graph[frm].values())
                    prob = self.graph[frm][to] / total if total > 0 else 0
                
                if prob >= (1 - ratio):
                    filtered.append(to)
            
            if len(filtered) < len(seq) * ratio and len(seq) > 3:
                step = max(2, int(1 / ratio))
                filtered = seq[::step]
            
            compressed.append(filtered)
        
        orig_len = sum(len(s) for s in sequences)
        comp_len = sum(len(s) for s in compressed)
        print(f"  Compression: {orig_len} → {comp_len} actions ({comp_len/orig_len*100:.1f}%)")
        
        return compressed
    
    def save_outputs(self, output_dir: str):
        """Save all GSS outputs"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save transition matrix
        np.save(f"{output_dir}/transition_matrix.npy", self.transition_matrix)
        
        # Save JSON hints
        with open(f"{output_dir}/json_hints.json", "w") as f:
            json.dump(self.json_hints, f, indent=2)
        
        # Save semantic clusters
        with open(f"{output_dir}/semantic_clusters.pkl", "wb") as f:
            pickle.dump(self.semantic_clusters, f)
        
        print(f"\nGSS outputs saved to: {output_dir}")


def run_gss_complete(dataset: str, ori_env: str, actions_dict: Dict[str, int],
                     sequences: List[List[int]]) -> Dict:
    """
    One-function call to run complete GSS
    """
    print("\n" + "="*60)
    print("SMARTGEN GSS - COMPLETE")
    print("="*60)
    
    gss = SmartGenGSS(actions_dict)
    gss.build_graph(sequences)
    gss.extract_matrix(len(actions_dict))
    gss.semantic_mapping()
    gss.generate_hints()
    compressed = gss.compress(sequences)
    
    output_dir = f"IoT_data/{dataset}/{ori_env}/gss_output"
    gss.save_outputs(output_dir)
    
    # Save compressed sequences
    with open(f"{output_dir}/compressed_sequences.pkl", "wb") as f:
        pickle.dump(compressed, f)
    
    return {
        'graph': gss.graph,
        'transition_matrix': gss.transition_matrix,
        'semantic_clusters': gss.semantic_clusters,
        'json_hints': gss.json_hints,
        'compressed_sequences': compressed
    }