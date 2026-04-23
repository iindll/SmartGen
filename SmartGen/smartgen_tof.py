"""
smartgen_tof.py - Complete TOF Module for SmartGen
Add this file to your SmartGen folder - works with existing code
"""

import pickle
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from sklearn.ensemble import IsolationForest


class SmartGenTOF:
    """
    Complete TOF with:
    - Stage 1: Outlier Detection (Isolation Forest)
    - Stage 2: Valuable Selection (5 criteria + diversity)
    """
    
    def __init__(self, contamination: float = 0.1, min_value: float = 0.4):
        self.contamination = contamination
        self.min_value = min_value
        self.stage1_model = None
        self.action_frequencies = {}
        
        # Stage 2 weights
        self.weights = {
            'frequency': 0.25,      # How common are these actions?
            'completeness': 0.20,   # Is the sequence well-formed?
            'diversity': 0.15,      # Are actions varied?
            'predictability': 0.20, # Are transitions predictable?
            'coherence': 0.20       # Semantic coherence?
        }
        
        self.selection_history = []  # For diversity boost
    
    def fit(self, sequences: List[List[int]], transition_matrix: np.ndarray = None):
        """Train TOF on normal sequences"""
        print("\n[TOF] Training Two-stage Filter...")
        
        # Build action frequencies
        self.action_frequencies = defaultdict(int)
        for seq in sequences:
            for a in seq:
                self.action_frequencies[a] += 1
        
        # Train Stage 1: Isolation Forest
        features = self._extract_features(sequences, transition_matrix)
        self.stage1_model = IsolationForest(
            contamination=self.contamination, 
            random_state=42,
            n_estimators=100
        )
        self.stage1_model.fit(features)
        
        print(f"  Stage 1: Isolation Forest trained")
        print(f"  Stage 2: Value threshold = {self.min_value}")
        
        return self
    
    def _extract_features(self, sequences: List[List[int]], 
                          transition_matrix: np.ndarray) -> np.ndarray:
        """Extract features for outlier detection"""
        features = []
        
        for seq in sequences:
            feat = [
                len(seq),  # length
                len(set(seq)) / max(len(seq), 1),  # uniqueness
            ]
            
            # Add transition probabilities
            if transition_matrix is not None and len(seq) > 1:
                probs = []
                for i in range(len(seq) - 1):
                    frm, to = seq[i], seq[i+1]
                    if frm < transition_matrix.shape[0] and to < transition_matrix.shape[1]:
                        probs.append(transition_matrix[frm, to])
                feat.append(np.mean(probs) if probs else 0)
                feat.append(np.std(probs) if len(probs) > 1 else 0)
            else:
                feat.extend([0.5, 0])
            
            features.append(feat)
        
        return np.array(features)
    
    def stage1_predict(self, sequence: List[int], 
                       transition_matrix: np.ndarray) -> Tuple[bool, float]:
        """Stage 1: Is this sequence an outlier?"""
        features = self._extract_features([sequence], transition_matrix)
        
        if self.stage1_model:
            pred = self.stage1_model.predict(features)[0]
            score = self.stage1_model.score_samples(features)[0]
            is_outlier = (pred == -1)
            confidence = 1 / (1 + np.exp(-score))
        else:
            is_outlier = False
            confidence = 0.5
        
        return is_outlier, confidence
    
    def stage2_score(self, sequence: List[int], 
                     semantic_clusters: Dict[int, int] = None,
                     transition_matrix: np.ndarray = None) -> float:
        """Stage 2: Calculate value score (0-1)"""
        scores = {}
        
        # 1. Frequency score
        if self.action_frequencies:
            total = sum(self.action_frequencies.values())
            if total > 0:
                avg_freq = np.mean([self.action_frequencies.get(a, 0) / total for a in sequence])
                scores['frequency'] = min(avg_freq * 10, 1.0)
            else:
                scores['frequency'] = 0.5
        else:
            scores['frequency'] = 0.5
        
        # 2. Completeness (good length)
        ideal_len = 8
        scores['completeness'] = min(len(sequence) / ideal_len, 1.0)
        
        # 3. Diversity (variety of actions)
        scores['diversity'] = len(set(sequence)) / max(len(sequence), 1)
        
        # 4. Predictability (transition probabilities)
        if transition_matrix is not None and len(sequence) > 1:
            probs = []
            for i in range(len(sequence) - 1):
                frm, to = sequence[i], sequence[i+1]
                if frm < transition_matrix.shape[0] and to < transition_matrix.shape[1]:
                    probs.append(transition_matrix[frm, to])
            scores['predictability'] = np.mean(probs) if probs else 0.5
        else:
            scores['predictability'] = 0.5
        
        # 5. Semantic coherence
        if semantic_clusters and len(sequence) > 1:
            clusters = [semantic_clusters.get(a, -1) for a in sequence]
            changes = sum(1 for i in range(len(clusters)-1) if clusters[i] != clusters[i+1])
            scores['coherence'] = 1 - (changes / max(len(clusters)-1, 1))
        else:
            scores['coherence'] = 0.5
        
        # Weighted sum
        total = sum(scores[k] * self.weights[k] for k in scores)
        return total
    
    def stage2_select(self, sequences: List[List[int]], 
                      scores: List[float] = None,
                      semantic_clusters: Dict[int, int] = None,
                      transition_matrix: np.ndarray = None,
                      max_keep: int = None) -> List[List[int]]:
        """Stage 2: Select valuable sequences"""
        
        if scores is None:
            scores = [self.stage2_score(seq, semantic_clusters, transition_matrix) 
                     for seq in sequences]
        
        # Filter by min value
        candidates = [(seq, score) for seq, score in zip(sequences, scores) 
                     if score >= self.min_value]
        
        if not candidates:
            return []
        
        # Apply diversity boost
        boosted = []
        for seq, score in candidates:
            # Calculate max similarity to already selected
            max_sim = 0
            for hist_seq in self.selection_history[-20:]:
                sim = len(set(seq) & set(hist_seq)) / max(len(set(seq) | set(hist_seq)), 1)
                max_sim = max(max_sim, sim)
            
            diversity_bonus = (1 - max_sim) * 0.2
            boosted.append((seq, min(score + diversity_bonus, 1.0)))
        
        # Sort by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)
        
        if max_keep and len(boosted) > max_keep:
            boosted = boosted[:max_keep]
        
        # Update history
        self.selection_history.extend([seq for seq, _ in boosted])
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-100:]
        
        return [seq for seq, _ in boosted]
    
    def filter(self, sequences: List[List[int]],
               semantic_clusters: Dict[int, int] = None,
               transition_matrix: np.ndarray = None,
               max_sequences: int = None) -> List[List[int]]:
        """Apply complete two-stage filtering"""
        
        print("\n" + "="*50)
        print("TOF: Two-stage Filtering")
        print("="*50)
        print(f"  Input: {len(sequences)} sequences")
        
        # STAGE 1: Outlier Detection
        print("\n  [Stage 1] Outlier Detection...")
        stage1_passed = []
        for seq in sequences:
            is_outlier, _ = self.stage1_predict(seq, transition_matrix)
            if not is_outlier:
                stage1_passed.append(seq)
        
        print(f"    Passed: {len(stage1_passed)}/{len(sequences)}")
        print(f"    Removed: {len(sequences) - len(stage1_passed)} outliers")
        
        if not stage1_passed:
            return []
        
        # STAGE 2: Valuable Selection
        print("\n  [Stage 2] Valuable Selection...")
        scores = [self.stage2_score(seq, semantic_clusters, transition_matrix) 
                 for seq in stage1_passed]
        
        print(f"    Score range: {min(scores):.3f} - {max(scores):.3f}")
        
        selected = self.stage2_select(stage1_passed, scores, semantic_clusters, 
                                      transition_matrix, max_sequences)
        
        print(f"    Selected: {len(selected)} sequences")
        
       
        print("\n" + "-"*40)
        print("TOF SUMMARY")
        print("-"*40)
        print(f"  Input: {len(sequences)}")
        print(f"  Stage 1 removed: {len(sequences) - len(stage1_passed)}")
        print(f"  Stage 2 removed: {len(stage1_passed) - len(selected)}")
        print(f"  Final output: {len(selected)}")
        print("-"*40)
        
        return selected
    
    def save(self, filepath: str):
        """Save TOF model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'stage1_model': self.stage1_model,
                'weights': self.weights,
                'min_value': self.min_value,
                'action_frequencies': dict(self.action_frequencies)
            }, f)
        print(f"\nTOF model saved to: {filepath}")


def run_tof_complete(dataset: str, new_env: str,
                     compressed_sequences: List[List[int]],
                     semantic_clusters: Dict[int, int],
                     transition_matrix: np.ndarray) -> List[List[int]]:
    """
    One-function call to run complete TOF
    """
    print("\n" + "="*60)
    print("SMARTGEN TOF - COMPLETE")
    print("="*60)
    
    tof = SmartGenTOF(contamination=0.1, min_value=0.4)
    tof.fit(compressed_sequences, transition_matrix)
    
    filtered = tof.filter(
        sequences=compressed_sequences,
        semantic_clusters=semantic_clusters,
        transition_matrix=transition_matrix
    )
    
    # Save results
    output_dir = f"filter_data/{dataset}/{new_env}/tof_output"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    tof.save(f"{output_dir}/tof_model.pkl")
    
    with open(f"{output_dir}/filtered_sequences.pkl", "wb") as f:
        pickle.dump(filtered, f)
    
    print(f"\nFiltered sequences saved to: {output_dir}")
    
    return filtered