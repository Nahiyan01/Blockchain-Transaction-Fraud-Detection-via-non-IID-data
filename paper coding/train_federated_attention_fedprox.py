"""
Stabilizing Tabular Attention via FedProx Proximal Regularization
Research Sprint Implementation: Testing Hypothesis H3 (Overturning Hasan et al., 2026 Transformer Collapse)

Combines:
  1. Multi-Head Tabular Self-Attention + LayerNorm
  2. FedProx Proximal Regularization (mu > 0) to damp multi-head attention weight oscillation under alpha = 0.01.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def run_attention_fedprox_experiment(rounds=25, alpha=0.01, num_clients=20, mu=0.01, lr=0.01):
    print("=" * 80, flush=True)
    print(f" STABILIZING TABULAR ATTENTION VIA FEDPROX (mu={mu}, alpha={alpha})", flush=True)
    print("=" * 80, flush=True)
    
    out_dir = "paper coding/results/tabular_attention_fedprox"
    os.makedirs(out_dir, exist_ok=True)
    
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
    
    # 2. Partition
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
        
    global_model = create_stabilized_tabular_attention(input_dim=20, embed_dim=16, num_heads=2)
    loc_model = create_stabilized_tabular_attention(input_dim=20, embed_dim=16, num_heads=2)
    global_weights = global_model.get_weights()
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0)
    
    history = {
        "round": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }
    
    t_start = time.time()
    
    for r in range(1, rounds + 1):
        client_weights = []
        client_samples = []
        
        # Convert global weights to tensors for proximal term
        global_tensors = [tf.convert_to_tensor(w, dtype=tf.float32) for w in global_weights]
        
        for k in range(num_clients):
            k_idx = client_indices[k]
            if len(k_idx) == 0:
                continue
            X_k = tf.cast(X_train_scaled[k_idx], tf.float32)
            y_k = tf.cast(y_train[k_idx], tf.float32)
            N_k = len(X_k)
            
            loc_model.set_weights(global_weights)
            
            # 1 local epoch using batches
            batch_size = 32
            indices = np.random.permutation(N_k)
            for start_idx in range(0, N_k, batch_size):
                b_idx = indices[start_idx : start_idx + batch_size]
                bx = tf.gather(X_k, b_idx)
                by = tf.gather(y_k, b_idx)
                
                # Sample weights
                sw = by * 2.258 + (1.0 - by) * 0.642
                
                with tf.GradientTape() as tape:
                    preds = loc_model(bx, training=True)
                    preds = tf.squeeze(preds, axis=-1)
                    eps = 1e-7
                    p_clip = tf.clip_by_value(preds, eps, 1.0 - eps)
                    bce = -(sw * (by * tf.math.log(p_clip) + (1.0 - by) * tf.math.log(1.0 - p_clip)))
                    loss = tf.reduce_mean(bce)
                    
                    # Proximal term
                    prox = 0.0
                    for w_loc, w_g in zip(loc_model.trainable_variables, global_tensors):
                        prox += tf.reduce_sum(tf.square(w_loc - w_g))
                        
                    total_loss = loss + (mu / 2.0) * prox
                    
                grads = tape.gradient(total_loss, loc_model.trainable_variables)
                grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
                optimizer.apply_gradients(zip(grads, loc_model.trainable_variables))
                
            client_weights.append(loc_model.get_weights())
            client_samples.append(N_k)
            
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
            print(f"[Attn + FedProx] Round {r:02d}/{rounds:02d} -> Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%", flush=True)

    t_elapsed = time.time() - t_start
    y_prob_final = global_model.predict(X_test_scaled, verbose=0).ravel()
    y_pred_final = (y_prob_final >= 0.5).astype(int)
    
    final_acc = float(accuracy_score(y_test, y_pred_final))
    final_prec = float(precision_score(y_test, y_pred_final, zero_division=0))
    final_rec = float(recall_score(y_test, y_pred_final, zero_division=0))
    final_f1 = float(f1_score(y_test, y_pred_final, zero_division=0))
    final_auc = float(roc_auc_score(y_test, y_prob_final))
    
    res = {
        "architecture": "Tabular Attention + FedProx",
        "mu": mu,
        "communication_rounds": rounds,
        "runtime_seconds": round(t_elapsed, 2),
        "accuracy": final_acc,
        "precision": final_prec,
        "recall": final_rec,
        "f1_score": final_f1,
        "roc_auc": final_auc,
        "history": history
    }
    
    with open(os.path.join(out_dir, "attn_fedprox_metrics.json"), "w") as f:
        json.dump(res, f, indent=4)
        
    global_model.save(os.path.join(out_dir, "attn_fedprox_model.keras"))
    
    print("\n" + "=" * 80, flush=True)
    print(f" ATTN + FEDPROX RESULT: Acc = {final_acc*100:.2f}% | F1 = {final_f1*100:.2f}% | ROC-AUC = {final_auc:.4f}", flush=True)
    print("=" * 80, flush=True)
    return res

if __name__ == "__main__":
    run_attention_fedprox_experiment(rounds=25, alpha=0.01, num_clients=20, mu=0.05, lr=0.01)
