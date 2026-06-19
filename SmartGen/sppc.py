import os
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from models1 import TimeSeriesDataset1  

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
def setup_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Padding
def pad(vocab_size, sequences, max_len=40):
    for seq in sequences:
        if len(seq) < max_len:
            seq.extend([vocab_size - 1] * (max_len - len(seq)))
    return sequences

# Data Loader
def make_data(vocab_size, data_file, batch_size):
    with open(data_file, 'rb') as f:
        sequences = pickle.load(f)

    sequences = pad(vocab_size, sequences)
    sequences = np.array(sequences)

    dataset = TimeSeriesDataset1(vocab_size, sequences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model (Encoder + Projection)
class TransformerEncoderCL(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, proj_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )

    def forward(self, src, padding_mask=None):
        x = self.embedding(src)
        memory = self.encoder(x, src_key_padding_mask=padding_mask)

        pooled = memory.mean(dim=1)   # [B, D]
        z = self.projection(pooled)   # [B, proj_dim]

        return z

# Augmentations
def augment_sequence(seq, vocab_size, mask_prob=0.1, drop_prob=0.1):
    seq = seq.clone()

    # Masking
    mask = torch.rand(seq.shape, device=seq.device) < mask_prob
    seq[mask] = vocab_size - 1

    # Drop tokens
    keep = torch.rand(seq.shape, device=seq.device) > drop_prob
    seq = seq * keep

    return seq

def create_views(src, vocab_size):
    v1 = augment_sequence(src, vocab_size)
    v2 = augment_sequence(src, vocab_size)
    return v1, v2

# InfoNCE Loss (SimCLR)
def info_nce_loss(z1, z2, temperature=0.5):
    B = z1.shape[0]

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    sim = torch.matmul(z, z.T)

    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    sim = sim / temperature

    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])

    loss = F.cross_entropy(sim, labels)
    return loss

# Clustering Selection
def cluster_select(embeddings, sequences, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(embeddings)

    selected_indices = []

    for cid in range(num_clusters):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue

        centroid = kmeans.cluster_centers_[cid]
        cluster_emb = embeddings[idx]

        distances = np.linalg.norm(cluster_emb - centroid, axis=1)
        best_idx = idx[np.argmin(distances)]

        selected_indices.append(best_idx)

    return [sequences[i] for i in selected_indices]

# Main Compression Function
def CLUSTER_select(dataset, ori_env, vocab_size, num_clusters=50, epochs=5):
    setup_seed()

    for day in range(7):
        print(f"\n Processing Day {day}")

        file_path = f'IoT_data/{dataset}/{ori_env}/trn_day_{day}.pkl'

        with open(file_path, 'rb') as f:
            text_collection = pickle.load(f)

        loader = make_data(vocab_size, file_path, batch_size=len(text_collection))

        model = TransformerEncoderCL(vocab_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    
        # Train (Contrastive)
        model.train()
        for epoch in range(epochs):
            for src, padding_mask, _ in loader:
                src = src.to(device)

                v1, v2 = create_views(src, vocab_size)

                z1 = model(v1, padding_mask)
                z2 = model(v2, padding_mask)

                loss = info_nce_loss(z1, z2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    
        # Inference (NO AUGMENT)
        model.eval()
        with torch.no_grad():
            for src, padding_mask, _ in loader:
                src = src.to(device)
                z = model(src, padding_mask)
                embeddings = z.cpu().numpy()

        
        # Clustering Compression
        selected = cluster_select(embeddings, text_collection, num_clusters)

        print(f"Original: {len(text_collection)} → Compressed: {len(selected)}")

        save_path = f'IoT_data/{dataset}/{ori_env}/trn_day_{day}_CLUSTER.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(selected, f)
