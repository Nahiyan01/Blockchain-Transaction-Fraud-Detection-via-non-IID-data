"""
High-Performance Personalized Federated Learning (FedPer) Simulation Engine
Blockchain Fraud Detection under Extreme Non-IID Dirichlet Skew (alpha = 0.01)

Mathematical Formulation:
  - Base Representation Layers (Shared Globally via FedAvg):
      W1 in R^(20 x 64), b1 in R^64, W2 in R^(64 x 32), b2 in R^32
      Aggregated each round: W_base^(t+1) = sum_k (n_k / N) * W_base^(k)
      
  - Personalized Classification Heads (Private to Each Client Node):
      W3^(k) in R^(32 x 1), b3^(k) in R^1
      Preserved locally on node k across communication rounds.
      Never shared with central server (Zero transmission overhead).
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf

class FedPerNode:
    def __init__(self, in_dim=20, h1=64, h2=32, out_dim=1, seed=42):
        rng = np.random.RandomState(seed)
        # Shared base parameters (will be overwritten by global base)
        self.W1 = rng.randn(in_dim, h1).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = rng.randn(h1, h2).astype(np.float32) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2, dtype=np.float32)
        # Private local classification head
        self.W3 = rng.randn(h2, out_dim).astype(np.float32) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

    def get_base_weights(self):
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]

    def set_base_weights(self, base_weights):
        self.W1 = base_weights[0].copy()
        self.b1 = base_weights[1].copy()
        self.W2 = base_weights[2].copy()
        self.b2 = base_weights[3].copy()

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0.0, z2)
        z3 = a2 @ self.W3 + self.b3
        p = 1.0 / (1.0 + np.exp(-np.clip(z3, -15.0, 15.0)))
        return z1, a1, z2, a2, z3, p

    def predict_proba(self, X):
        _, _, _, _, _, p = self.forward(X)
        return p.ravel()

    def train_local(self, X_k, y_k, epochs=2, batch_size=32, lr=0.02, l2_reg=1e-4, w0=0.642, w1=2.258):
        N = len(X_k)
        if N == 0:
            return
        y_k = y_k.reshape(-1, 1).astype(np.float32)
        
        for epoch in range(epochs):
            indices = np.random.permutation(N)
            for start_idx in range(0, N, batch_size):
                b_idx = indices[start_idx : start_idx + batch_size]
                X_b = X_k[b_idx]
                y_b = y_k[b_idx]
                B = len(X_b)
                
                # Forward pass
                z1 = X_b @ self.W1 + self.b1
                a1 = np.maximum(0.0, z1)
                z2 = a1 @ self.W2 + self.b2
                a2 = np.maximum(0.0, z2)
                z3 = a2 @ self.W3 + self.b3
                p = 1.0 / (1.0 + np.exp(-np.clip(z3, -15.0, 15.0)))
                
                # Cost-sensitive weights
                sample_w = y_b * w1 + (1.0 - y_b) * w0
                dz3 = sample_w * (p - y_b) / B
                
                # Gradient of Private Head
                dW3 = a2.T @ dz3 + l2_reg * self.W3
                db3 = np.sum(dz3, axis=0)
                
                # Backpropagation to Base Representation
                da2 = dz3 @ self.W3.T
                dz2 = da2 * (z2 > 0.0)
                dW2 = a1.T @ dz2 + l2_reg * self.W2
                db2 = np.sum(dz2, axis=0)
                
                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0.0)
                dW1 = X_b.T @ dz1 + l2_reg * self.W1
                db1 = np.sum(dz1, axis=0)
                
                # Gradient clipping
                for g in [dW1, dW2, dW3]:
                    norm = np.linalg.norm(g)
                    if norm > 1.0:
                        g /= norm
                        
                # Update both Base and Private Head
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W3 -= lr * dW3
                self.b3 -= lr * db3

def aggregate_base(base_weights_list, sample_sizes):
    total = sum(sample_sizes)
    agg = []
    for w_idx in range(len(base_weights_list[0])):
        w_sum = np.zeros_like(base_weights_list[0][w_idx])
        for k in range(len(base_weights_list)):
            w_sum += (sample_sizes[k] / total) * base_weights_list[k][w_idx]
        agg.append(w_sum)
    return agg

def run_fedper_benchmark(rounds=50, local_epochs=2, num_clients=20, alpha=0.01, lr=0.02):
    print("=" * 80, flush=True)
    print(f" HIGH-SPEED PERSONALIZED FL (FedPer) SIMULATION (alpha = {alpha})", flush=True)
    print("=" * 80, flush=True)
    
    out_dir = "paper coding/results/fedper_personalized"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load and de-skew dataset
    df = pd.read_csv("paper coding/processed_data/cleaned_ethereum_20features.csv")
    X = df.drop(columns=['FLAG']).copy()
    y = df['FLAG'].values
    
    for col in X.columns:
        if X[col].min() >= 0:
            X[col] = np.log1p(X[col])
            
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Non-IID Dirichlet Partition
    np.random.seed(42)
    num_classes = 2
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        idx_c = np.where(y_train == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        client_splits = np.split(idx_c, splits)
        for k in range(num_clients):
            client_indices[k].extend(client_splits[k].tolist())
            
    for k in range(num_clients):
        np.random.shuffle(client_indices[k])
        
    # Split each client's partition into 80% local train and 20% local test to measure personalization!
    client_train_data = []
    client_test_data = []
    
    for k in range(num_clients):
        k_idx = client_indices[k]
        if len(k_idx) > 10:
            tr, te = train_test_split(k_idx, test_size=0.2, random_state=42)
        else:
            tr = k_idx
            te = k_idx
        client_train_data.append((X_train_scaled[tr], y_train[tr]))
        client_test_data.append((X_train_scaled[te], y_train[te]))
        
    # Initialize 20 Client Nodes with personalized heads
    clients = [FedPerNode(in_dim=20, h1=64, h2=32, out_dim=1, seed=42 + k) for k in range(num_clients)]
    global_base_weights = clients[0].get_base_weights()
    
    history = {
        "round": [],
        "avg_client_accuracy": [],
        "avg_client_f1": [],
        "avg_client_recall": [],
        "global_holdout_accuracy": [],
        "global_holdout_f1": []
    }
    
    t_start = time.time()
    
    for r in range(1, rounds + 1):
        client_base_weights = []
        client_sample_sizes = []
        
        # 1. Local Training
        for k in range(num_clients):
            X_k, y_k = client_train_data[k]
            if len(X_k) == 0:
                continue
                
            # Synchronize base representation with global base
            clients[k].set_base_weights(global_base_weights)
            
            # Local training updates base AND personalized head
            clients[k].train_local(X_k, y_k, epochs=local_epochs, batch_size=32, lr=lr)
            
            client_base_weights.append(clients[k].get_base_weights())
            client_sample_sizes.append(len(X_k))
            
        # 2. Server FedAvg on Base Layers ONLY
        global_base_weights = aggregate_base(client_base_weights, client_sample_sizes)
        
        # 3. Synchronize global base back to evaluate local test sets
        local_accs, local_f1s, local_recs = [], [], []
        for k in range(num_clients):
            X_te_k, y_te_k = client_test_data[k]
            if len(X_te_k) == 0:
                continue
            clients[k].set_base_weights(global_base_weights)
            p_k = clients[k].predict_proba(X_te_k)
            pred_k = (p_k >= 0.5).astype(int)
            local_accs.append(accuracy_score(y_te_k, pred_k))
            local_f1s.append(f1_score(y_te_k, pred_k, zero_division=0))
            local_recs.append(recall_score(y_te_k, pred_k, zero_division=0))
            
        mean_acc = float(np.mean(local_accs))
        mean_f1 = float(np.mean(local_f1s))
        mean_rec = float(np.mean(local_recs))
        
        # 4. Global Holdout Evaluation via Ensemble of Personalized Heads
        ensemble_p = np.zeros(len(X_test_scaled))
        valid_count = 0
        for k in range(num_clients):
            if len(client_train_data[k][0]) > 0:
                ensemble_p += clients[k].predict_proba(X_test_scaled)
                valid_count += 1
        ensemble_p /= max(1, valid_count)
        ensemble_pred = (ensemble_p >= 0.5).astype(int)
        
        glob_acc = float(accuracy_score(y_test, ensemble_pred))
        glob_f1 = float(f1_score(y_test, ensemble_pred, zero_division=0))
        
        history["round"].append(r)
        history["avg_client_accuracy"].append(mean_acc)
        history["avg_client_f1"].append(mean_f1)
        history["avg_client_recall"].append(mean_rec)
        history["global_holdout_accuracy"].append(glob_acc)
        history["global_holdout_f1"].append(glob_f1)
        
        if r % 10 == 0 or r == 1:
            print(f"Round {r:02d}/50 | Avg Local F1: {mean_f1*100:.2f}% | Local Rec: {mean_rec*100:.2f}% | Global Test F1: {glob_f1*100:.2f}% | Global Acc: {glob_acc*100:.2f}%", flush=True)

    t_elapsed = time.time() - t_start
    final_auc = float(roc_auc_score(y_test, ensemble_p))
    cm = confusion_matrix(y_test, ensemble_pred).tolist()
    
    fedper_summary = {
        "framework": "Personalized Federated Learning (FedPer)",
        "alpha": alpha,
        "num_clients": num_clients,
        "communication_rounds": rounds,
        "runtime_seconds": round(t_elapsed, 2),
        "final_avg_local_accuracy": mean_acc,
        "final_avg_local_f1": mean_f1,
        "final_avg_local_recall": mean_rec,
        "final_global_test_accuracy": glob_acc,
        "final_global_test_f1": glob_f1,
        "final_global_test_roc_auc": final_auc,
        "confusion_matrix": cm,
        "history": history
    }
    
    with open(os.path.join(out_dir, "fedper_metrics.json"), "w") as f:
        json.dump(fedper_summary, f, indent=4)
        
    # Plot FedPer Results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["round"], [x * 100 for x in history["avg_client_f1"]], label="Avg Local Client F1 (%)", lw=2.2, color="#2ca02c")
    plt.plot(history["round"], [x * 100 for x in history["global_holdout_f1"]], label="Global Ensemble Test F1 (%)", lw=2.2, color="#1f77b4", linestyle="--")
    plt.title("FedPer: Local Personalization vs Global Test F1", fontsize=12, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=10)
    plt.ylabel("F1-Score (%)", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["round"], [x * 100 for x in history["avg_client_recall"]], label="Avg Local Client Recall (%)", lw=2.2, color="#d62728")
    plt.plot(history["round"], [x * 100 for x in history["avg_client_accuracy"]], label="Avg Local Client Accuracy (%)", lw=2.2, color="#9467bd")
    plt.title("FedPer: Client-Level Recall & Accuracy", fontsize=12, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=10)
    plt.ylabel("Percentage (%)", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "fedper_convergence.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("\n" + "=" * 80, flush=True)
    print(f" FEDPER SIMULATION COMPLETED IN {t_elapsed:.2f}s!", flush=True)
    print(f"  Avg Local Client F1: {mean_f1*100:.2f}% | Avg Local Recall: {mean_rec*100:.2f}% | Global ROC-AUC: {final_auc:.4f}", flush=True)
    print(f"  Plot saved to: {plot_path}", flush=True)
    print("=" * 80, flush=True)
    return fedper_summary

if __name__ == "__main__":
    run_fedper_benchmark(rounds=50, local_epochs=2, num_clients=20, alpha=0.01, lr=0.02)
