"""
High-Performance FedProx Simulation Engine
Blockchain Fraud Detection under Extreme Non-IID Dirichlet Skew (alpha = 0.01)

Mathematical Formulation:
  Local Objective on Client k:
    min_w  F_k(w) + (mu / 2) * ||w - w^t||_2^2
  Where:
    - F_k(w) is cost-sensitive weighted binary cross-entropy with L2 weight decay
    - (mu / 2) * ||w - w^t||_2^2 is the FedProx proximal operator
    - mu = 0.0 reduces identically to standard FedAvg
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
from tensorflow.keras import layers, models, regularizers

# 1. 2NN Vectorized Parameter Container
class TwoLayerNN:
    def __init__(self, in_dim=20, h1=64, h2=32, out_dim=1, seed=42):
        rng = np.random.RandomState(seed)
        # He initialization
        self.W1 = rng.randn(in_dim, h1).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = rng.randn(h1, h2).astype(np.float32) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W3 = rng.randn(h2, out_dim).astype(np.float32) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

    def get_weights(self):
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(), self.W3.copy(), self.b3.copy()]

    def set_weights(self, weights):
        self.W1 = weights[0].copy()
        self.b1 = weights[1].copy()
        self.W2 = weights[2].copy()
        self.b2 = weights[3].copy()
        self.W3 = weights[4].copy()
        self.b3 = weights[5].copy()

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

# 2. Local FedProx Training on Client k
def train_client_fedprox_numpy(
    model, global_weights, X_k, y_k, epochs=2, batch_size=32, lr=0.02, mu=0.01,
    l2_reg=1e-4, w0=0.642, w1=2.258
):
    N = len(X_k)
    if N == 0:
        return
    y_k = y_k.reshape(-1, 1).astype(np.float32)
    
    # Global reference weights for proximal term
    W1_g, b1_g, W2_g, b2_g, W3_g, b3_g = global_weights
    
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        for start_idx in range(0, N, batch_size):
            b_idx = indices[start_idx : start_idx + batch_size]
            X_b = X_k[b_idx]
            y_b = y_k[b_idx]
            B = len(X_b)
            
            # Forward pass
            z1 = X_b @ model.W1 + model.b1
            a1 = np.maximum(0.0, z1)
            z2 = a1 @ model.W2 + model.b2
            a2 = np.maximum(0.0, z2)
            z3 = a2 @ model.W3 + model.b3
            p = 1.0 / (1.0 + np.exp(-np.clip(z3, -15.0, 15.0)))
            
            # Cost-sensitive weights: w_i = w1 if y=1 else w0
            sample_w = y_b * w1 + (1.0 - y_b) * w0
            
            # Weighted BCE gradient: dL/dz3 = sample_w * (p - y)
            dz3 = sample_w * (p - y_b) / B
            
            # Layer 3 Gradients + Proximal term
            dW3 = a2.T @ dz3 + l2_reg * model.W3 + mu * (model.W3 - W3_g)
            db3 = np.sum(dz3, axis=0) + mu * (model.b3 - b3_g)
            
            # Layer 2 Backpropagation
            da2 = dz3 @ model.W3.T
            dz2 = da2 * (z2 > 0.0)
            dW2 = a1.T @ dz2 + l2_reg * model.W2 + mu * (model.W2 - W2_g)
            db2 = np.sum(dz2, axis=0) + mu * (model.b2 - b2_g)
            
            # Layer 1 Backpropagation
            da1 = dz2 @ model.W2.T
            dz1 = da1 * (z1 > 0.0)
            dW1 = X_b.T @ dz1 + l2_reg * model.W1 + mu * (model.W1 - W1_g)
            db1 = np.sum(dz1, axis=0) + mu * (model.b1 - b1_g)
            
            # Gradient clipping
            for g in [dW1, dW2, dW3]:
                norm = np.linalg.norm(g)
                if norm > 1.0:
                    g /= norm
                    
            # Parameter updates
            model.W1 -= lr * dW1
            model.b1 -= lr * db1
            model.W2 -= lr * dW2
            model.b2 -= lr * db2
            model.W3 -= lr * dW3
            model.b3 -= lr * db3

# 3. Server FedAvg Aggregation
def aggregate_weights(client_weights_list, client_sample_sizes):
    total = sum(client_sample_sizes)
    agg = []
    for w_idx in range(len(client_weights_list[0])):
        w_sum = np.zeros_like(client_weights_list[0][w_idx])
        for k in range(len(client_weights_list)):
            w_sum += (client_sample_sizes[k] / total) * client_weights_list[k][w_idx]
        agg.append(w_sum)
    return agg

# 4. Convert NumPy weights to Keras model for artifact persistence
def export_to_keras(weights, input_dim=20):
    m = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])
    m.set_weights(weights)
    m.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return m

# 5. Full Benchmark Runner
def run_fedprox_benchmark(mu_values=[0.0, 0.001, 0.005, 0.01, 0.05], rounds=50, local_epochs=2, lr=0.02):
    print("=" * 80, flush=True)
    print(" HIGH-PERFORMANCE FEDPROX BENCHMARK (Dirichlet alpha = 0.01)", flush=True)
    print("=" * 80, flush=True)
    
    out_dir = "paper coding/results/fedprox_benchmarks"
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
    
    # 20 Clients Dirichlet non-IID partition (alpha = 0.01)
    np.random.seed(42)
    num_clients = 20
    alpha = 0.01
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
        
    benchmark_results = {}
    
    for mu in mu_values:
        label = "FedAvg (mu=0.0)" if mu == 0.0 else f"FedProx (mu={mu})"
        print(f"\n---> Running: {label} (50 Rounds, 20 Clients, alpha=0.01)...", flush=True)
        t_start = time.time()
        
        global_model = TwoLayerNN(in_dim=20, h1=64, h2=32, out_dim=1, seed=42)
        global_weights = global_model.get_weights()
        
        # Local worker model
        local_model = TwoLayerNN(in_dim=20, h1=64, h2=32, out_dim=1, seed=42)
        
        history = {
            "round": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": []
        }
        
        for r in range(1, rounds + 1):
            client_weights = []
            client_samples = []
            
            for k in range(num_clients):
                k_idx = client_indices[k]
                if len(k_idx) == 0:
                    continue
                X_k = X_train_scaled[k_idx]
                y_k = y_train[k_idx]
                
                # Reset local model to current global weights
                local_model.set_weights(global_weights)
                
                # Run local FedProx optimization
                train_client_fedprox_numpy(
                    local_model, global_weights, X_k, y_k,
                    epochs=local_epochs, batch_size=32, lr=lr, mu=mu
                )
                
                client_weights.append(local_model.get_weights())
                client_samples.append(len(X_k))
                
            # Aggregate weights
            global_weights = aggregate_weights(client_weights, client_samples)
            global_model.set_weights(global_weights)
            
            # Predict on server test set
            y_prob = global_model.predict_proba(X_test_scaled)
            y_pred = (y_prob >= 0.5).astype(int)
            
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            
            history["round"].append(r)
            history["accuracy"].append(acc)
            history["precision"].append(prec)
            history["recall"].append(rec)
            history["f1_score"].append(f1)
            
            if r % 10 == 0 or r == 1:
                print(f"  [{label}] Round {r:02d}/50 | Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%", flush=True)
                
        t_elapsed = time.time() - t_start
        y_prob_final = global_model.predict_proba(X_test_scaled)
        y_pred_final = (y_prob_final >= 0.5).astype(int)
        
        final_acc = float(accuracy_score(y_test, y_pred_final))
        final_prec = float(precision_score(y_test, y_pred_final, zero_division=0))
        final_rec = float(recall_score(y_test, y_pred_final, zero_division=0))
        final_f1 = float(f1_score(y_test, y_pred_final, zero_division=0))
        final_auc = float(roc_auc_score(y_test, y_prob_final))
        cm = confusion_matrix(y_test, y_pred_final).tolist()
        
        benchmark_results[str(mu)] = {
            "label": label,
            "mu": mu,
            "training_time_sec": round(t_elapsed, 2),
            "accuracy": final_acc,
            "precision": final_prec,
            "recall": final_rec,
            "f1_score": final_f1,
            "roc_auc": final_auc,
            "confusion_matrix": cm,
            "history": history
        }
        
        # Save Keras format artifact
        k_model = export_to_keras(global_weights, input_dim=20)
        k_model.save(os.path.join(out_dir, f"model_fedprox_mu_{mu}.keras"))
        print(f"  ---> COMPLETED in {t_elapsed:.2f}s: Acc={final_acc*100:.2f}%, Prec={final_prec*100:.2f}%, Rec={final_rec*100:.2f}%, F1={final_f1*100:.2f}%, AUC={final_auc:.4f}", flush=True)

    # Save complete JSON
    with open(os.path.join(out_dir, "fedprox_comparison_metrics.json"), "w") as f:
        json.dump(benchmark_results, f, indent=4)
        
    # Generate publication-grade comparative plot
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    for mu_str, res in benchmark_results.items():
        plt.plot(res["history"]["round"], [x * 100 for x in res["history"]["f1_score"]], label=res["label"], lw=2.2)
    plt.title("FedProx vs FedAvg: F1-Score Convergence (alpha=0.01)", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Test F1-Score (%)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for mu_str, res in benchmark_results.items():
        plt.plot(res["history"]["round"], [x * 100 for x in res["history"]["recall"]], label=res["label"], lw=2.2)
    plt.title("FedProx vs FedAvg: Recall Recovery under Client Drift", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Test Recall (%)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "fedprox_vs_fedavg_convergence.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("\n" + "=" * 80, flush=True)
    print(" ALL FEDPROX BENCHMARKS FINISHED. Plot saved to:", plot_path, flush=True)
    print("=" * 80, flush=True)
    return benchmark_results

if __name__ == "__main__":
    run_fedprox_benchmark(mu_values=[0.0, 0.001, 0.005, 0.01, 0.05], rounds=50, local_epochs=2, lr=0.02)
