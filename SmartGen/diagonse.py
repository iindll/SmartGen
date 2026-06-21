import pickle
import numpy as np
import torch
from baseline1 import (
    setup_seed, vocab_dic, collect_embeddings, _load_model,
    pad, split_random
)
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
from models1 import TimeSeriesDataset2

dataset = "us"
new_env = "spring"
method = "SPPC"
thres = 0.918
model_name = "gpt-4o"

setup_seed(2024)
vocab_size = vocab_dic[dataset]
model_path = f"check_model/contrastive_{dataset}_{model_name}_{method}.pth"

train_file = f'IoT_data/{dataset}/{new_env}/trn.pkl'
vld_file   = f'IoT_data/{dataset}/{new_env}/rs_vld.pkl'
test_file1 = f"attack/{dataset}/labeled_{dataset}_spring_attack_heater.pkl"
test_file3 = f"IoT_data/{dataset}/{new_env}/split_test.pkl"

print("=" * 60)
print("Loading embeddings...")
print("=" * 60)

train_emb = collect_embeddings(new_env, vocab_size, train_file, model_path)
val_emb   = collect_embeddings(new_env, vocab_size, vld_file, model_path)
normal_emb = np.vstack([train_emb, val_emb])
print(f"Normal embeddings: {normal_emb.shape}")

with open(test_file1, 'rb') as f:
    attack_data = pickle.load(f)
with open(test_file3, 'rb') as f:
    tmp = pickle.load(f)
    tmp = pad(vocab_size, tmp)
    normal_test_data = [(item, 0) for item in tmp]

all_data = normal_test_data + attack_data
sequences = [item[0] for item in all_data]
pad(vocab_size, sequences)
labels = np.array([item[1] for item in all_data])
sequences = np.array(sequences)

test_dataset = TimeSeriesDataset2(vocab_size, sequences)
loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model = _load_model(model_path, vocab_size)

all_emb = []
with torch.no_grad():
    for src, pmask, _ in loader:
        emb = model.encode(src, pmask)
        all_emb.append(emb.cpu().numpy())
test_emb = np.vstack(all_emb)

print(f"Test embeddings: {test_emb.shape}  (labels: {labels.shape})")

print("\n" + "=" * 60)
print("Analysis 1: Embedding Norm Statistics")
print("=" * 60)
normal_test_emb = test_emb[labels == 0]
attack_emb = test_emb[labels == 1]

print(f"Normal (train) embedding norm  — mean: {np.linalg.norm(normal_emb, axis=1).mean():.4f}")
print(f"Normal (test)  embedding norm  — mean: {np.linalg.norm(normal_test_emb, axis=1).mean():.4f}")
print(f"Attack (test)  embedding norm  — mean: {np.linalg.norm(attack_emb, axis=1).mean():.4f}")

print("\n" + "=" * 60)
print("Analysis 2: Centroid Distances")
print("=" * 60)
normal_centroid = normal_emb.mean(axis=0)
normal_test_centroid = normal_test_emb.mean(axis=0)
attack_centroid = attack_emb.mean(axis=0)

print(f"Distance (train normal centroid -> test normal centroid): "
      f"{np.linalg.norm(normal_centroid - normal_test_centroid):.4f}")
print(f"Distance (train normal centroid -> attack centroid):      "
      f"{np.linalg.norm(normal_centroid - attack_centroid):.4f}")
print(f"Distance (test normal centroid  -> attack centroid):      "
      f"{np.linalg.norm(normal_test_centroid - attack_centroid):.4f}")

print("\n" + "=" * 60)
print("Analysis 3: F1 Score across different nu values")
print("=" * 60)

scaler = StandardScaler()
X_train = scaler.fit_transform(normal_emb)
X_test = scaler.transform(test_emb)

best_f1 = 0
best_nu = None
for nu in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    ocsvm.fit(X_train)
    raw_preds = ocsvm.predict(X_test)
    preds = [1 if p == -1 else 0 for p in raw_preds]
    f1 = f1_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    print(f"  nu={nu:<5} F1={f1:.4f}  Recall={rec:.4f}  Precision={prec:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_nu = nu

print(f"\nBest nu = {best_nu}  (F1={best_f1:.4f})")

print("\n" + "=" * 60)
print(f"Analysis 4: F1 Score across different gamma (nu={best_nu})")
print("=" * 60)

for gamma in ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]:
    ocsvm = OneClassSVM(kernel='rbf', nu=best_nu, gamma=gamma)
    ocsvm.fit(X_train)
    raw_preds = ocsvm.predict(X_test)
    preds = [1 if p == -1 else 0 for p in raw_preds]
    f1 = f1_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    print(f"  gamma={str(gamma):<8} F1={f1:.4f}  Recall={rec:.4f}  Precision={prec:.4f}")

print("\nDone.")
