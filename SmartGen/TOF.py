import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SmartGenTOF_Improved:    
    def __init__(self, contamination: float = 0.02, min_value: float = 0.3, 
                 strict_mode: bool = False):
        self.contamination = contamination 
        self.min_value = min_value
        self.strict_mode = strict_mode
        self.stage1_model = None
        self.action_frequencies = {}
        self.sequence_stats = {} 
        self.weights = {
            'frequency': 0.25,
            'completeness': 0.20,
            'diversity': 0.15,
            'predictability': 0.20,
            'coherence': 0.20
        }
        self.selection_history = []
        self.is_fitted = False
    
    def _normalize_sequence(self, sequence: List[int]) -> List[int]:
        if not sequence:
            return []
        valid_seq = [int(a) for a in sequence 
                    if isinstance(a, (int, np.integer)) and 0 <= a < 10000]
        return valid_seq if valid_seq else []
    
    def _is_valid_sequence(self, sequence: List[int]) -> bool:
        if not sequence:
            return False
        if len(sequence) < 2:
            return False
        if len(sequence) > 100:
            return False
        return True
    
    def _extract_features(self, sequences: List[List[int]], 
                          transition_matrix: Optional[np.ndarray]) -> np.ndarray:
        features = []
        for seq in sequences:
            try:
                valid_seq = self._normalize_sequence(seq)
                if not valid_seq or len(valid_seq) < 2:
                    continue
                
                length_feat = min(len(valid_seq), 100) / 100.0
                unique_feat = len(set(valid_seq)) / max(len(valid_seq), 1)
                mean_transition = 0.5
                std_transition = 0.0
                
                if transition_matrix is not None and len(valid_seq) > 1:
                    try:
                        if (isinstance(transition_matrix, np.ndarray) and
                            transition_matrix.shape[0] > 0):
                            probs = []
                            for i in range(min(len(valid_seq) - 1, 50)):
                                frm, to = valid_seq[i], valid_seq[i+1]
                                if (0 <= frm < transition_matrix.shape[0] and
                                    0 <= to < transition_matrix.shape[1]):
                                    val = transition_matrix[frm, to]
                                    if isinstance(val, (int, float, np.number)):
                                        if np.isfinite(val):
                                            probs.append(float(val))
                            if probs:
                                mean_transition = np.mean(probs)
                                std_transition = np.std(probs) if len(probs) > 1 else 0.0
                    except:
                        pass
                
                feat = [float(length_feat), float(unique_feat), 
                        float(mean_transition), float(std_transition)]
                features.append(feat)
            except Exception as e:
                continue
        
        return np.array(features, dtype=float) if features else np.zeros((0, 4), dtype=float)
    
    def fit(self, sequences: List[List[int]], 
            transition_matrix: Optional[np.ndarray] = None):
        print("\n[TOF] Training Two-stage Filter...")
        
        valid_sequences = []
        for seq in sequences:
            norm_seq = self._normalize_sequence(seq)
            if self._is_valid_sequence(norm_seq):
                valid_sequences.append(norm_seq)
        
        print(f"  Pre-filter: {len(sequences)} → {len(valid_sequences)} valid sequences")
        
        if not valid_sequences or len(valid_sequences) < 2:
            print("  WARNING: Too few valid sequences")
            self.is_fitted = False
            return self
        
        self.action_frequencies = defaultdict(int)
        for seq in valid_sequences:
            for a in seq:
                self.action_frequencies[int(a)] += 1
        
        features = self._extract_features(valid_sequences, transition_matrix)
        
        if features.shape[0] < 2:
            print("  WARNING: Too few sequences for training")
            self.is_fitted = False
            return self
        
        actual_contamination = min(self.contamination, 0.5)
        if features.shape[0] < 5:
            actual_contamination = 0.0
        elif features.shape[0] < 10:
            actual_contamination = 0.05
        
        try:
            self.stage1_model = IsolationForest(
                contamination=actual_contamination,
                random_state=42,
                n_estimators=min(100, features.shape[0]),
                max_samples='auto'
            )
            self.stage1_model.fit(features)
            self.is_fitted = True
            
            print(f"  ✓ Stage 1: Isolation Forest trained")
            print(f"  ✓ Stage 2: Value threshold = {self.min_value}")
            print(f"  ✓ Mode: {'STRICT' if self.strict_mode else 'LENIENT'}")
        except Exception as e:
            print(f"  ERROR in fit: {str(e)}")
            self.is_fitted = False
        
        return self
    
    def stage1_predict(self, sequence: List[int]) -> Tuple[bool, float]:
        try:
            if not self.is_fitted or self.stage1_model is None:
                return False, 0.5
            
            norm_seq = self._normalize_sequence(sequence)
            if not norm_seq or len(norm_seq) < 2:
                return True, 1.0
            
            features = self._extract_features([norm_seq], None)
            if features.shape[0] == 0:
                return True, 1.0
            
            pred = self.stage1_model.predict(features)[0]
            is_outlier = (pred == -1)
            
            if not self.strict_mode and is_outlier:
                return False, 0.5
            
            return bool(is_outlier), 0.5
        except Exception:
            return False, 0.5
    
    def stage2_score(self, sequence: List[int]) -> float:
        try:
            norm_seq = self._normalize_sequence(sequence)
            if not norm_seq or len(norm_seq) < 2:
                return 0.0
            
            scores = {}
            
            if self.action_frequencies:
                total = sum(self.action_frequencies.values())
                if total > 0:
                    freqs = [self.action_frequencies.get(a, 1) / total for a in norm_seq]
                    avg_freq = np.mean(freqs) if freqs else 0.5
                    scores['frequency'] = min(avg_freq * 10, 1.0)
                else:
                    scores['frequency'] = 0.5
            else:
                scores['frequency'] = 0.5
            
            if 2 <= len(norm_seq) <= 100:
                scores['completeness'] = 1.0
            elif len(norm_seq) == 1:
                scores['completeness'] = 0.5
            else:
                scores['completeness'] = 0.7
            
            diversity_score = len(set(norm_seq)) / max(len(norm_seq), 1)
            scores['diversity'] = min(diversity_score, 1.0)
            scores['predictability'] = 0.5
            scores['coherence'] = 0.5
            
            total = sum(scores[k] * self.weights[k] for k in scores if k in self.weights)
            return float(np.clip(total, 0.0, 1.0))
        except Exception:
            return 0.5
    
    def filter(self, sequences: List[List[int]],
               semantic_clusters: Optional[Dict[int, int]] = None,
               transition_matrix: Optional[np.ndarray] = None,
               max_sequences: Optional[int] = None) -> List[List[int]]:
        print("\n" + "="*50)
        print("TOF: Two-stage Filtering")
        print("="*50)
        
        if not sequences:
            print("  ERROR: No sequences provided")
            return []
        
        print(f"  Input: {len(sequences)} sequences")
        
        valid_seqs = []
        invalid_count = 0
        
        for idx, seq in enumerate(sequences):
            norm_seq = self._normalize_sequence(seq)
            if self._is_valid_sequence(norm_seq):
                valid_seqs.append(norm_seq)
            else:
                invalid_count += 1
        
        print(f"    ✓ Valid: {len(valid_seqs)}/{len(sequences)}")
        print(f"    ✗ Corrupted: {invalid_count}")
        
        if not valid_seqs:
            return []
        
        if not self.is_fitted:
            self.fit(valid_seqs, transition_matrix)
        
        stage1_passed = []
        outlier_count = 0
        
        for seq in valid_seqs:
            is_outlier, _ = self.stage1_predict(seq)
            if not is_outlier:
                stage1_passed.append(seq)
            else:
                outlier_count += 1
        
        print(f"    ✓ Passed: {len(stage1_passed)}/{len(valid_seqs)}")
        print(f"    ✗ Removed: {outlier_count}")
        
        if not stage1_passed:
            print("  WARNING: No sequences passed Stage 1, returning valid sequences")
            return valid_seqs
        
        scores = [self.stage2_score(seq) for seq in stage1_passed]
        
        if scores:
            print(f"    Score range: {min(scores):.3f} - {max(scores):.3f}")
        
        selected = [(seq, score) for seq, score in zip(stage1_passed, scores) 
                   if score >= self.min_value]
        
        if not selected:
            print(f"    WARNING: No sequences above threshold, keeping top sequences")
            threshold_seqs = sorted(zip(stage1_passed, scores), 
                                   key=lambda x: x[1], reverse=True)
            keep_count = max(1, len(threshold_seqs) // 2)
            selected = threshold_seqs[:keep_count]
        
        selected_seqs = [seq for seq, _ in selected]
        print(f"    ✓ Selected: {len(selected_seqs)} sequences")
        
        print("\n" + "-"*40)
        print("TOF SUMMARY")
        print("-"*40)
        print(f"  Input:          {len(sequences)}")
        print(f"  Corrupted:      {invalid_count}")
        print(f"  Valid sequences:{len(valid_seqs)}")
        print(f"  Stage 1 kept:   {len(stage1_passed)}")
        print(f"  Stage 2 kept:   {len(selected_seqs)}")
        print(f"  Final output:   {len(selected_seqs)}")
        
        valid_rate = len(selected_seqs) / len(sequences) * 100 if sequences else 0
        print(f"  Valid rate:     {valid_rate:.1f}%")
        print("-"*40)
        
        return selected_seqs


# ===== INTEGRATION FUNCTION - what main.py calls =====
def filter_generated_sequences(dataset, new_env, threshold, method, model, 
                               contamination=0.05, min_value=0.4):
    """
    Standalone function to filter generated sequences with TOF.
    main.py only needs to call this one function.
    """
    import os
    import pickle
    import shutil
    
    print("\n" + "="*60)
    print("  TOF Filter Integration")
    print("="*60)
    
    base_name = f'{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq'
    input_file = f'filter_data/{dataset}/{new_env}/{base_name}.pkl'
    output_file = f'filter_data/{dataset}/{new_env}/{base_name}_tof.pkl'
    target_file = f'filter_data/{dataset}/{new_env}/{base_name}_filter_true.pkl'
    
    try:
        with open(input_file, 'rb') as f:
            sequences = pickle.load(f)
        print(f"  Loaded {len(sequences)} sequences from {input_file}")
        
        if not sequences:
            print("  ERROR: No sequences loaded!")
            return False
        
        tof = SmartGenTOF_Improved(contamination=contamination, min_value=min_value, strict_mode=False)
        filtered_sequences = tof.filter(sequences)
        
        with open(output_file, 'wb') as f:
            pickle.dump(filtered_sequences, f)
        
        shutil.copy2(output_file, target_file)
        
        print(f"\n  ✅ TOF Filtering Complete!")
        print(f"  Input: {len(sequences)} sequences")
        print(f"  Output: {len(filtered_sequences)} sequences")
        print(f"  Retention: {len(filtered_sequences)/len(sequences)*100:.1f}%")
        print(f"  Saved to: {output_file}")
        print("="*60)
        
        return True
        
    except FileNotFoundError:
        print(f"  ERROR: Input file not found: {input_file}")
        print("  Skipping TOF filtering...")
        return False
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False
