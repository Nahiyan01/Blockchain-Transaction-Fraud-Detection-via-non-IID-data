"""
High-Speed Federated Training Simulation for Stabilized Tabular Attention
Research Sprint Implementation: Overcoming Transformer Collapse in Non-IID Blockchain Fraud Detection

Tests whether LayerNorm + Residual Pre-LN + Feature Tokenization prevents the 72.88% collapse
reported in Hasan et al. (2026) under extreme Dirichlet skew (alpha = 0.01).
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

from models_attention import create_stabilized_tabular_attention

def fed_avg(client_weights_list, client_sample_sizes):
    total = sum(client_sample_sizes)
    agg_weights = []
    for w_idx in range(len(client_weights_list[0])):
        w_sum = np.zeros_like(client_weights_list[0][w_idx])
        for k in range(len(client_weights_list)):
            w_sum += (client_sample_sizes[k] / total) * client_weights_list[k][w_idx]
        agg_weights.append(w_sum)
    return agg_weights

def run_attention_federated_experiment(rounds=30, local_epochs=1, alpha=0.01, num_clients=20):
    print("=" * 80, flush=True)
    print(f" FEDERATED TABULAR SELF-ATTENTION SIMULATION (Dirichlet alpha = {alpha})", flush=True)
    print("=" * 80, flush=True)
    
    output_dir = "paper coding/results/tabular_attention_federated"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and de-skew dataset
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
    
    # 2. Partition across 20 clients with Dirichlet distribution (alpha = 0.01)
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
        
    # 3. Initialize Global and Worker Attention Models (Reusable to avoid graph bloat)
    print("\nInitializing Stabilized Tabular Attention Networks...", flush=True)
    global_model = create_stabilized_tabular_attention(input_dim=20, embed_dim=16, num_heads=2)
    loc_model = create_stabilized_tabular_attention(input_dim=20, embed_dim=16, num_heads=2)
    global_weights = global_model.get_weights()
    
    history = {
        "round": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }
    
    print(f"Executing {rounds} Communication Rounds (20 Clients, {local_epochs} local epochs/round)...\n", flush=True)
    t_start = time.time()
    
    for r in range(1, rounds + 1):
        client_weights = []
        client_samples = []
        
        for k in range(num_clients):
            k_idx = client_indices[k]
            if len(k_idx) == 0:
                continue
            X_k = X_train_scaled[k_idx]
            y_k = y_train[k_idx]
            
            # Re-use worker model
            loc_model.set_weights(global_weights)
            
            loc_model.fit(
                X_k, y_k,
                epochs=local_epochs,
                batch_size=32,
                class_weight={0: 0.642, 1: 2.258},
                verbose=0
            )
            
            client_weights.append(loc_model.get_weights())
            client_samples.append(len(X_k))
            
        # FedAvg Aggregation
        global_weights = fed_avg(client_weights, client_samples)
        global_model.set_weights(global_weights)
        
        # Server Evaluation
        y_prob = global_model.predict(X_test_scaled, verbose=0).ravel()
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
        
        if r % 5 == 0 or r == 1:
            print(f"[Tabular Attention] Round {r:02d}/{rounds:02d} -> Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%", flush=True)

    t_elapsed = time.time() - t_start
    # Final evaluation
    y_prob_final = global_model.predict(X_test_scaled, verbose=0).ravel()
    y_pred_final = (y_prob_final >= 0.5).astype(int)
    
    final_acc = float(accuracy_score(y_test, y_pred_final))
    final_prec = float(precision_score(y_test, y_pred_final, zero_division=0))
    final_rec = float(recall_score(y_test, y_pred_final, zero_division=0))
    final_f1 = float(f1_score(y_test, y_pred_final, zero_division=0))
    final_auc = float(roc_auc_score(y_test, y_prob_final))
    cm = confusion_matrix(y_test, y_pred_final).tolist()
    
    attention_results = {
        "architecture": "Stabilized Tabular Self-Attention",
        "framework": "Federated Learning (FedAvg)",
        "alpha": alpha,
        "num_clients": num_clients,
        "communication_rounds": rounds,
        "runtime_seconds": round(t_elapsed, 2),
        "accuracy": final_acc,
        "precision": final_prec,
        "recall": final_rec,
        "f1_score": final_f1,
        "roc_auc": final_auc,
        "confusion_matrix": cm,
        "history": history
    }
    
    with open(os.path.join(output_dir, "tabular_attention_metrics.json"), "w") as f:
        json.dump(attention_results, f, indent=4)
        
    global_model.save(os.path.join(output_dir, "tabular_attention_model.keras"))
    
    # Plot Convergence Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history["round"], [x * 100 for x in history["accuracy"]], label="Our Stabilized Tabular Attention Accuracy (%)", lw=2.2, color="#1f77b4")
    plt.plot(history["round"], [x * 100 for x in history["f1_score"]], label="Our Stabilized Attention F1-Score (%)", lw=2.2, color="#ff7f0e")
    plt.axhline(y=72.88, color='r', linestyle='--', label="Hasan et al. TabTF Baseline (72.88% Collapse)")
    plt.title(f"Rescuing Transformers from Non-IID Collapse (alpha={alpha})", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Metric (%)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "tabular_attention_convergence.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("\n" + "=" * 80, flush=True)
    print(f" TABULAR ATTENTION RESULT: Acc = {final_acc*100:.2f}% | F1 = {final_f1*100:.2f}% | ROC-AUC = {final_auc:.4f} (Finished in {t_elapsed:.2f}s)", flush=True)
    print("=" * 80, flush=True)
    return attention_results

if __name__ == "__main__":
    run_attention_federated_experiment(rounds=25, local_epochs=1, alpha=0.01, num_clients=20)
