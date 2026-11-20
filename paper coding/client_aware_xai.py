"""
Client-Aware Explainable AI (XAI) Drift Audit
Research Sprint Implementation: Testing Hypothesis H4 (Multi-Client Feature Attribution Drift)

In real-world decentralized financial systems, retail wallets and crypto exchanges have fundamentally
different transaction behaviors. This script audits local feature attribution drift across diverse
client nodes under Dirichlet Non-IID skew (alpha = 0.01).
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import tensorflow as tf

def run_client_aware_xai():
    print("=" * 80, flush=True)
    print(" CLIENT-AWARE EXPLAINABLE AI (XAI) DRIFT AUDIT", flush=True)
    print("=" * 80, flush=True)
    
    out_dir = "paper coding/results/explainability_client_aware"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load data & metadata
    df = pd.read_csv("paper coding/processed_data/cleaned_ethereum_20features.csv")
    X = df.drop(columns=['FLAG']).copy()
    y = df['FLAG'].values
    
    with open("paper coding/processed_data/dataset_summary.json") as f:
        meta = json.load(f)
    feature_names = meta["features"]
    
    for col in X.columns:
        if X[col].min() >= 0:
            X[col] = np.log1p(X[col])
            
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Partition across 20 clients with alpha=0.01
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
        
    # Analyze client label distributions to find 3 distinct client archetypes
    client_profiles = []
    for k in range(num_clients):
        k_y = y_train[client_indices[k]]
        total = len(k_y)
        if total == 0:
            continue
        fraud_cnt = int(np.sum(k_y == 1))
        fraud_pct = fraud_cnt / total * 100.0
        client_profiles.append((k, total, fraud_cnt, fraud_pct))
        
    # Sort by fraud percentage
    client_profiles.sort(key=lambda x: x[3])
    
    # Pick Node A (Normal-dominant, 0% fraud), Node B (Fraud-dominant, high fraud), Node C (Moderate)
    node_normal = client_profiles[0]   # Lowest fraud
    node_fraud = client_profiles[-1]   # Highest fraud
    node_mid = client_profiles[len(client_profiles) // 2]  # Median
    
    selected_nodes = [
        {"id": node_normal[0], "role": "Retail Wallet Node (Normal-Dominant)", "fraud_pct": node_normal[3], "n": node_normal[1]},
        {"id": node_mid[0], "role": "Crypto Exchange Node (Mixed-Volume)", "fraud_pct": node_mid[3], "n": node_mid[1]},
        {"id": node_fraud[0], "role": "Sybil / Phishing Cluster (Fraud-Dominant)", "fraud_pct": node_fraud[3], "n": node_fraud[1]}
    ]
    
    print("\nSelected Archetype Client Nodes for XAI Audit:")
    for n in selected_nodes:
        print(f"  • Node {n['id']:02d}: {n['role']} | Total={n['n']} | Fraud={n['fraud_pct']:.1f}%")
        
    # 3. Load Global Optimized Model
    model_path = "paper coding/results/optimized_centralized/optimized_centralized_2nn.keras"
    if not os.path.exists(model_path):
        model_path = "paper coding/results/centralized/centralized_2nn_model.keras"
    print(f"\nLoading model from: {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    def predict_fn(x):
        return model.predict(x, verbose=0).ravel()
        
    # Use 30 background samples
    bg_data = shap.kmeans(X_train_scaled, 25)
    explainer = shap.KernelExplainer(predict_fn, bg_data)
    
    node_shap_results = {}
    top_n = 7
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    
    for i, node in enumerate(selected_nodes):
        k_idx = client_indices[node["id"]]
        # Evaluate up to 40 samples from this client
        sample_size = min(40, len(k_idx))
        sample_indices = k_idx[:sample_size]
        X_node = X_train_scaled[sample_indices]
        
        print(f"Computing SHAP values for Node {node['id']:02d} ({sample_size} samples)...", flush=True)
        shap_vals = explainer.shap_values(X_node, nsamples=100)
        
        # Mean absolute SHAP per feature
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
        
        top_features = [feature_names[idx] for idx in top_indices]
        top_importances = [float(mean_abs_shap[idx]) for idx in top_indices]
        
        node_shap_results[f"node_{node['id']}"] = {
            "role": node["role"],
            "fraud_percentage": node["fraud_pct"],
            "top_features": top_features,
            "top_importances": top_importances
        }
        
        # Plot horizontal bar chart
        ax = axes[i]
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_importances[::-1], color=colors[i], alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=10)
        ax.set_title(f"Node {node['id']:02d}: {node['role']}\n(Fraud Ratio: {node['fraud_pct']:.1f}%)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.4)
        
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "client_aware_xai_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Save JSON summary
    with open(os.path.join(out_dir, "client_aware_xai_summary.json"), "w") as f:
        json.dump(node_shap_results, f, indent=4)
        
    print("\n" + "=" * 80, flush=True)
    print(" CLIENT-AWARE XAI DRIFT AUDIT COMPLETED!", flush=True)
    print(f"  Visualization saved to: {plot_path}", flush=True)
    print("=" * 80, flush=True)
    return node_shap_results

if __name__ == "__main__":
    run_client_aware_xai()
