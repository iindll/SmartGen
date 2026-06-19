import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, confusion_matrix)
from torch.utils.data import DataLoader

from models1 import TimeSeriesDataset2, TimeSeriesDataset3, TimeSeriesDataset4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_dic = {"an": 141, "fr": 223, "us": 269, "sp": 235}


# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════

def setup_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def pad(vocab_size, sequences):
    for i in range(len(sequences)):
        if len(sequences[i]) < 40:
            sequences[i].extend([vocab_size - 1] * (40 - len(sequences[i])))
        elif len(sequences[i]) > 40:
            sequences[i] = sequences[i][:40]
    return sequences


def split_random(data_file, train_file, vld_file, split_ratio=0.8, seed=2024):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    random.seed(seed)
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    ensure_dir_exists(train_file)
    ensure_dir_exists(vld_file)
    with open(train_file, 'wb') as f:
        pickle.dump(data[:split_index], f)
    with open(vld_file, 'wb') as f:
        pickle.dump(data[split_index:], f)
    return data[:split_index], data[split_index:]


def make_data(new_env, vocab_size, data_file, batch_size=32, shuffle=False):
    with open(data_file, 'rb') as f:
        sequences = pickle.load(f)
    data = pad(vocab_size, sequences)
    data = np.array(data)
    if new_env == 'spring':
        dataset = TimeSeriesDataset2(vocab_size, data)
    elif new_env == 'night':
        dataset = TimeSeriesDataset3(vocab_size, data)
    elif new_env == 'multiple':
        dataset = TimeSeriesDataset4(vocab_size, data)
    else:
        raise ValueError(f"Unknown new_env: {new_env}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ══════════════════════════════════════════════════════════════
# Encoder
# ══════════════════════════════════════════════════════════════

class ContrastiveEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, proj_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, batch_first=True, dropout=0.1,
            dim_feedforward=d_model * 2
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, padding_mask=None):
        x = self.embedding(src)
        memory = self.encoder(x, src_key_padding_mask=padding_mask)

        # كل quadruple = [day, hour, device, action]
        # positions 2,3,6,7,10,11,... هي الـ device+action tokens
        # دول الأهم للـ anomaly detection
        B, L, D = memory.shape
        device_action_mask = torch.zeros(B, L, device=memory.device)
        for pos in range(2, L, 4):   # device token
            device_action_mask[:, pos] = 3.0   # وزن أعلى للـ device
        for pos in range(3, L, 4):   # action token
            device_action_mask[:, pos] = 3.0   # وزن أعلى للـ action
        for pos in range(0, L, 4):   # day token
            device_action_mask[:, pos] = 1.0
        for pos in range(1, L, 4):   # hour token
            device_action_mask[:, pos] = 1.0

        # بنطبق الـ padding mask
        if padding_mask is not None:
            valid = (~padding_mask).float()
            device_action_mask = device_action_mask * valid

        # Weighted pooling
        weights = device_action_mask.unsqueeze(-1)           # [B, L, 1]
        pooled = (memory * weights).sum(1) / weights.sum(1).clamp(min=1e-9)
        return pooled

    def forward(self, src, padding_mask=None):
        return self.projector(self.encode(src, padding_mask))


# ══════════════════════════════════════════════════════════════
# Augmentation
# ══════════════════════════════════════════════════════════════

def augment(src, vocab_size, mask_prob=0.2, crop_prob=0.1):
    src = src.clone()
    # Token masking
    mask = torch.rand(src.shape, device=src.device) < mask_prob
    src[mask] = vocab_size - 1
    # Random crop (zero out end)
    if crop_prob > 0 and random.random() < crop_prob:
        L = src.shape[1]
        crop_len = random.randint(1, max(1, L // 4))
        src[:, L - crop_len:] = vocab_size - 1
    return src


def simclr_loss(z1, z2, temperature=0.07):
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device)
    ])
    return F.cross_entropy(sim, labels)


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

def train_contrastive(new_env, vocab_size, train_file, model_save_path,
                      epochs=100, lr=1e-4, temperature=0.07):
    ensure_dir_exists(model_save_path)

    with open(train_file, 'rb') as f:
        all_seqs = pickle.load(f)

    # بنستخدم كل الـ data في batch واحد لو صغيرة
    batch_size = min(64, len(all_seqs))
    loader = make_data(new_env, vocab_size, train_file,
                       batch_size=batch_size, shuffle=True)

    model = ContrastiveEncoder(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup = 5
    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        t = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    best_loss = float('inf')
    print(f"\n── Contrastive Training ({'GPU' if device.type == 'cuda' else 'CPU'}) ──")
    print(f"   samples={len(all_seqs)}, batch={batch_size}, temp={temperature}, epochs={epochs}")

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            src, pmask, _ = batch
            src = src.to(device)
            pmask = pmask.to(device)
            v1 = augment(src, vocab_size)
            v2 = augment(src, vocab_size)
            z1 = model(v1, pmask)
            z2 = model(v2, pmask)
            loss = simclr_loss(z1, z2, temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        avg = total / len(loader)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), model_save_path)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:>2}/{epochs}] loss={avg:.4f}  best={best_loss:.4f}")

    print(f"✓ Saved → {model_save_path}  (best loss: {best_loss:.4f})\n")


# ══════════════════════════════════════════════════════════════
# Embeddings
# ══════════════════════════════════════════════════════════════

def _load_model(path, vocab_size):
    m = ContrastiveEncoder(vocab_size).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def collect_embeddings(new_env, vocab_size, data_file, model_path, batch_size=64):
    loader = make_data(new_env, vocab_size, data_file, batch_size=batch_size)
    model = _load_model(model_path, vocab_size)
    all_emb = []
    with torch.no_grad():
        for src, pmask, _ in loader:
            src = src.to(device)
            pmask = pmask.to(device)
            all_emb.append(model.encode(src, pmask).cpu().numpy())
    return np.vstack(all_emb)


# ══════════════════════════════════════════════════════════════
# One-Class SVM Detection
# ══════════════════════════════════════════════════════════════

def build_ocsvm(normal_emb, nu=0.1):
    """
    One-Class SVM على الـ normal embeddings.
    nu = الـ upper bound على الـ fraction of outliers في الـ training data.
    قيمة صغيرة = أكثر صرامة.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(normal_emb)

    ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    ocsvm.fit(X)

    # تحقق على الـ normal data نفسها
    preds = ocsvm.predict(X)
    normal_acc = (preds == 1).mean()
    print(f"\n── One-Class SVM ──────────────────────────────────────")
    print(f"  Normal samples    : {len(normal_emb)}")
    print(f"  nu                : {nu}")
    print(f"  Normal data accuracy: {normal_acc:.2%}  (المفروض قريب من {1-nu:.0%})")
    print(f"──────────────────────────────────────────────────────\n")

    return ocsvm, scaler


# ══════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate(new_env, vocab_size, test_file1, test_file3,
             model_path, ocsvm, scaler):
    with open(test_file1, 'rb') as f:
        attack_data = pickle.load(f)
    with open(test_file3, 'rb') as f:
        tmp = pickle.load(f)
        tmp = pad(vocab_size, tmp)
        normal_data = [(item, 0) for item in tmp]

    all_data = normal_data + attack_data
    sequences = [item[0] for item in all_data]
    pad(vocab_size, sequences)
    labels = [item[1] for item in all_data]
    sequences = np.array(sequences)

    if new_env == 'spring':
        test_dataset = TimeSeriesDataset2(vocab_size, sequences)
    elif new_env == 'night':
        test_dataset = TimeSeriesDataset3(vocab_size, sequences)
    elif new_env == 'multiple':
        test_dataset = TimeSeriesDataset4(vocab_size, sequences)

    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = _load_model(model_path, vocab_size)

    all_emb = []
    with torch.no_grad():
        for src, pmask, _ in loader:
            src = src.to(device)
            pmask = pmask.to(device)
            all_emb.append(model.encode(src, pmask).cpu().numpy())

    test_emb = np.vstack(all_emb)
    X_test = scaler.transform(test_emb)

    # OC-SVM: +1 = normal, -1 = anomaly
    raw_preds = ocsvm.predict(X_test)
    predictions = [1 if p == -1 else 0 for p in raw_preds]

    cm = confusion_matrix(y_true=labels, y_pred=predictions)
    TN, FP, FN, TP = cm.ravel()
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    recall    = recall_score(y_true=labels, y_pred=predictions, zero_division=0)
    precision = precision_score(y_true=labels, y_pred=predictions, zero_division=0)
    accuracy  = accuracy_score(y_true=labels, y_pred=predictions)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n── Results ───────────────────────────────────────────")
    print(f"  True Positive    : {TP}")
    print(f"  True Negative    : {TN}")
    print(f"  False Positive   : {FP}")
    print(f"  False Negative   : {FN}")
    print(f"  FPR              : {FPR:.4f}")
    print(f"  FNR              : {FNR:.4f}")
    print(f"  Recall           : {recall:.4f}")
    print(f"  Precision        : {precision:.4f}")
    print(f"  Accuracy         : {accuracy:.4f}")
    print(f"  F1 Score         : {f1:.4f}")
    print(f"──────────────────────────────────────────────────────")
    print("Finished Test")

    return TP, TN, FP, FN, FPR, FNR, recall, precision, accuracy, f1


# ══════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════

def Anomaly_detection(dataset, new_env, thres, method, model,
                      percentage, k=10, cl_epochs=50):
    setup_seed(2024)

    vocab_size = vocab_dic[dataset]
    model_path = f"check_model/contrastive_{dataset}_{model}_{method}.pth"
    ensure_dir_exists(model_path)

    data_file = (f'filter_data/{dataset}/{new_env}/'
                 f'{dataset}_{new_env}_generation_{method}_th={thres}_{model}_seq_filter_true.pkl')

    if new_env == 'multiple':
        train_file = vld_file = data_file
    else:
        # بنقسم الـ real test normal data لـ train/val
        # عشان الـ model يتعلم على نفس distribution الـ test data
        real_test_file = f'IoT_data/{dataset}/{new_env}/split_test.pkl'
        train_file = f'IoT_data/{dataset}/{new_env}/trn.pkl'
        vld_file   = f'IoT_data/{dataset}/{new_env}/rs_vld.pkl'
        split_random(real_test_file, train_file, vld_file, split_ratio=0.8)

    if new_env == 'spring':
        test_file1 = f"attack/{dataset}/labeled_{dataset}_spring_attack_heater.pkl"
    elif new_env == 'night':
        test_file1 = f"attack/{dataset}/labeled_{dataset}_night_attack_time.pkl"
    elif new_env == 'multiple':
        test_file1 = f"attack/{dataset}/labeled_{dataset}_multiple_attack_tv.pkl"

    test_file3 = f"IoT_data/{dataset}/{new_env}/split_test.pkl"

    # Step 1: Train
    print("=" * 55)
    print("  Step 1: Contrastive Training on Normal Data")
    print("=" * 55)
    train_contrastive(new_env, vocab_size, train_file, model_path, epochs=cl_epochs)

    # Step 2: Embeddings
    print("=" * 55)
    print("  Step 2: Extracting Normal Embeddings")
    print("=" * 55)
    train_emb = collect_embeddings(new_env, vocab_size, train_file, model_path)
    val_emb   = collect_embeddings(new_env, vocab_size, vld_file,   model_path)
    normal_emb = np.vstack([train_emb, val_emb])
    print(f"  Normal embeddings: {normal_emb.shape}")

    # Step 3: One-Class SVM
    print("=" * 55)
    print("  Step 3: Building One-Class SVM Detector")
    print("=" * 55)
    # nu في OC-SVM = نسبة الـ outliers المسموح بيها في الـ training
    # بنستخدم قيمة أكبر عشان الـ SVM يبقى أقل صرامة على الـ normal data
    nu = 0.2
    ocsvm, scaler = build_ocsvm(normal_emb, nu=nu)

    # Step 4: Evaluate
    print("=" * 55)
    print("  Step 4: Evaluation")
    print("=" * 55)
    return evaluate(new_env, vocab_size, test_file1, test_file3,
                    model_path, ocsvm, scaler)
