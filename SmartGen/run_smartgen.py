"""
run_smartgen.py - Complete pipeline using the 2 new files
"""

import pickle
from smartgen_gss import run_gss_complete
from smartgen_tof import run_tof_complete
from dictionary import fr_actions  # Change based on your dataset

# CONFIGURATION
DATASET = "fr"
ORI_ENV = "winter"
NEW_ENV = "spring"

print("="*60)
print("SMARTGEN: GSS + TOF Complete Pipeline")
print("="*60)

# Load your data
with open(f'IoT_data/{DATASET}/{ORI_ENV}/split_trn.pkl', 'rb') as f:
    sequences = pickle.load(f)

# Extract action sequences (same as your ATM)
indices = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
action_seqs = []
for seq in sequences:
    extracted = [seq[i] for i in indices if i < len(seq)]
    if extracted:
        action_seqs.append(extracted)

print(f"Loaded {len(action_seqs)} action sequences")

# RUN GSS
gss_results = run_gss_complete(DATASET, ORI_ENV, fr_actions, action_seqs)

# RUN TOF
filtered = run_tof_complete(
    DATASET, NEW_ENV,
    gss_results['compressed_sequences'],
    gss_results['semantic_clusters'],
    gss_results['transition_matrix']
)

print("\n" + "="*60)
print(f"COMPLETE! Filtered {len(filtered)} sequences")
print("="*60)