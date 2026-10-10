"""
Federated Learning Simulation Script
Paper: "Interpretable Federated Learning for Fraud Detection in Non-IID Blockchain Transactions"
Authors: Mahmudul Hasan et al. (Blockchain: Research and Applications)

Implements:
- Stage 2: Client Partitioning (20 Clients, Dirichlet Non-IID Skew with alpha = 0.01 and IID)
- Stage 3: Local Training (2NN model with SGD optimizer)
- Stage 4: FedAvg Aggregation across Communication Rounds
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import tensorflow as tf

from models import create_2nn_model

def dirichlet_partition(y_train: np.ndarray, num_clients: int = 20, alpha: float = 0.01, seed: int = 42):
    """
    Partitions training data across `num_clients` using a Dirichlet distribution over class labels.
    alpha = 0.01 corresponds to the extreme Non-IID heterogeneity tested in the paper.
    """
    np.random.seed(seed)
    num_classes = len(np.unique(y_train))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(y_train == c)[0]
        np.random.shuffle(idx_c)

        # Draw Dirichlet proportions for class c across all clients
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Split indices according to proportions
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        client_splits = np.split(idx_c, splits)

        for k in range(num_clients):
            client_indices[k].extend(client_splits[k].tolist())

    for k in range(num_clients):
        np.random.shuffle(client_indices[k])

    return client_indices

def iid_partition(y_train: np.ndarray, num_clients: int = 20, seed: int = 42):
    """
    Partitions training data uniformly (IID) across `num_clients`.
    """
    np.random.seed(seed)
    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    return np.array_split(indices, num_clients)

def federated_averaging(client_weights_list, client_sample_sizes):
    """
    Executes standard FedAvg:
    W_global = sum( (n_k / N_total) * W_k )
    """
    total_samples = sum(client_sample_sizes)
    new_weights = []

    for weight_idx in range(len(client_weights_list[0])):
        weighted_sum = np.zeros_like(client_weights_list[0][weight_idx])
        for k in range(len(client_weights_list)):
            fraction = client_sample_sizes[k] / total_samples
            weighted_sum += fraction * client_weights_list[k][weight_idx]
        new_weights.append(weighted_sum)

    return new_weights

def run_federated_training(
    num_clients: int = 20,
    rounds: int = 40,
    local_epochs: int = 2,
    batch_size: int = 32,
    alpha: float = 0.01,
    partition_type: str = 'dirichlet',
    optimizer_name: str = 'sgd',
    data_dir: str = "processed_data",
    output_dir: str = "results/federated"
):
    os.makedirs(output_dir, exist_ok=True)
    print("="*75)
    print(f" FEDERATED LEARNING SIMULATION ({num_clients} Clients | {partition_type.upper()} alpha={alpha} | {optimizer_name.upper()})")
    print("="*75)

    # 1. Load Data
    X_train = np.load(os.path.join(data_dir, "X_train_scaled.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test_scaled.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    with open(os.path.join(data_dir, "dataset_summary.json"), "r") as f:
        meta = json.load(f)
    class_weights = {int(k): float(v) for k, v in meta["class_weights"].items()}

    # 2. Stage 2: Client Partitioning
    print(f"\n[Stage 2] Partitioning training data into {num_clients} client nodes...")
    if partition_type == 'dirichlet':
        client_indices = dirichlet_partition(y_train, num_clients=num_clients, alpha=alpha)
    else:
        client_indices = iid_partition(y_train, num_clients=num_clients)

    # Calculate client class statistics
    client_stats = []
    print("\nClient Partition Overview:")
    for k in range(num_clients):
        k_indices = client_indices[k]
        k_labels = y_train[k_indices]
        n_total = len(k_labels)
        n_fraud = int(np.sum(k_labels == 1))
        n_normal = int(np.sum(k_labels == 0))
        fraud_ratio = (n_fraud / n_total * 100) if n_total > 0 else 0
        client_stats.append({
            "client_id": k + 1,
            "total_samples": n_total,
            "normal_samples": n_normal,
            "fraud_samples": n_fraud,
            "fraud_ratio_pct": fraud_ratio
        })
        print(f"   Node {k+1:02d}: {n_total:4d} samples | Normal: {n_normal:4d} | Fraud: {n_fraud:4d} ({fraud_ratio:5.1f}%)")

    # Plot client distribution
    plot_client_distribution(client_stats, os.path.join(output_dir, "client_data_distribution.png"))

    # 3. Stage 3 & 4: Federated Training & FedAvg
    print(f"\n[Stage 3 & 4] Initializing Global Model (2NN + {optimizer_name.upper()})...")
    global_model = create_2nn_model(input_dim=X_train.shape[1], learning_rate=0.03, optimizer_name=optimizer_name)
    global_weights = global_model.get_weights()

    history = {
        "round": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "loss": []
    }

    print(f"Beginning Federated Communication Rounds ({rounds} rounds, {local_epochs} local epochs/round)...\n")

    for r in range(1, rounds + 1):
        client_weights = []
        client_samples = []

        # Local Client Training
        for k in range(num_clients):
            k_idx = client_indices[k]
            if len(k_idx) == 0:
                continue

            X_k = X_train[k_idx]
            y_k = y_train[k_idx]

            # Build local model initialized with global weights
            local_model = create_2nn_model(input_dim=X_train.shape[1], learning_rate=0.03, optimizer_name=optimizer_name)
            local_model.set_weights(global_weights)

            # Local training step
            local_model.fit(
                X_k, y_k,
                epochs=local_epochs,
                batch_size=min(batch_size, len(X_k)),
                class_weight=class_weights,
                verbose=0
            )

            client_weights.append(local_model.get_weights())
            client_samples.append(len(X_k))

        # Central Aggregation: FedAvg
        global_weights = federated_averaging(client_weights, client_samples)
        global_model.set_weights(global_weights)

        # Global Evaluation on Test Set
        y_prob = global_model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        eval_res = global_model.evaluate(X_test, y_test, verbose=0)
        round_loss = float(eval_res[0])
        round_acc = float(accuracy_score(y_test, y_pred))
        round_prec = float(precision_score(y_test, y_pred, zero_division=0))
        round_rec = float(recall_score(y_test, y_pred, zero_division=0))
        round_f1 = float(f1_score(y_test, y_pred, zero_division=0))

        history["round"].append(r)
        history["loss"].append(round_loss)
        history["accuracy"].append(round_acc)
        history["precision"].append(round_prec)
        history["recall"].append(round_rec)
        history["f1_score"].append(round_f1)

        if r % 5 == 0 or r == 1 or r == rounds:
            print(f"Round {r:02d}/{rounds:02d} | Loss: {round_loss:.4f} | Acc: {round_acc*100:.2f}% | Prec: {round_prec*100:.2f}% | Rec: {round_rec*100:.2f}% | F1: {round_f1*100:.2f}%")

    # 4. Final Global Evaluation
    print("\n" + "="*50)
    print(" FINAL FEDERATED TEST EVALUATION:")
    y_prob = global_model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    final_acc = accuracy_score(y_test, y_pred)
    final_prec = precision_score(y_test, y_pred, zero_division=0)
    final_rec = recall_score(y_test, y_pred, zero_division=0)
    final_f1 = f1_score(y_test, y_pred, zero_division=0)
    final_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"   Accuracy:    {final_acc * 100:.2f}%")
    print(f"   Precision:   {final_prec * 100:.2f}%")
    print(f"   Recall:      {final_rec * 100:.2f}%")
    print(f"   F1-Score:    {final_f1 * 100:.2f}%")
    print(f"   ROC-AUC:     {final_auc:.4f}")
    print("="*50)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    # 5. Save Artifacts & Visualizations
    model_save_path = os.path.join(output_dir, "federated_global_2nn_sgd.keras")
    global_model.save(model_save_path)

    metrics_export = {
        "framework": "Federated Learning (FedAvg)",
        "model": "2NN",
        "optimizer": optimizer_name,
        "num_clients": num_clients,
        "partition_type": partition_type,
        "alpha": alpha,
        "communication_rounds": rounds,
        "local_epochs": local_epochs,
        "final_accuracy": float(final_acc),
        "final_precision": float(final_prec),
        "final_recall": float(final_rec),
        "final_f1_score": float(final_f1),
        "final_roc_auc": float(final_auc),
        "confusion_matrix": cm.tolist(),
        "history": history
    }
    with open(os.path.join(output_dir, "federated_metrics.json"), "w") as f:
        json.dump(metrics_export, f, indent=4)

    # Plot convergence & confusion matrix
    plot_federated_performance(history, cm, final_f1, os.path.join(output_dir, "federated_convergence.png"))
    print(f"\nAll federated results successfully saved to: {output_dir}")
    return metrics_export

def plot_client_distribution(client_stats, save_path):
    client_ids = [s["client_id"] for s in client_stats]
    normal_counts = [s["normal_samples"] for s in client_stats]
    fraud_counts = [s["fraud_samples"] for s in client_stats]

    plt.figure(figsize=(12, 5))
    bar_width = 0.6
    p1 = plt.bar(client_ids, normal_counts, bar_width, label='Normal (FLAG=0)', color='#2b5c8f')
    p2 = plt.bar(client_ids, fraud_counts, bar_width, bottom=normal_counts, label='Fraud (FLAG=1)', color='#d9534f')

    plt.title("Stage 2: Client Data Partitioning under Non-IID Dirichlet Skew (alpha=0.01)", fontsize=13, fontweight='bold')
    plt.xlabel("Simulated Edge Client Node ID", fontsize=11)
    plt.ylabel("Number of Transaction Samples", fontsize=11)
    plt.xticks(client_ids)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_federated_performance(history, cm, f1, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[0],
                xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    axes[0].set_title(f"Federated 2NN Confusion Matrix (20 Clients)\nFinal F1-Score: {f1*100:.2f}%", fontsize=12)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Convergence history
    axes[1].plot(history["round"], [a * 100 for a in history["accuracy"]], label='Accuracy (%)', color='#2ca02c', lw=2)
    axes[1].plot(history["round"], [f * 100 for f in history["f1_score"]], label='F1-Score (%)', color='#1f77b4', lw=2)
    axes[1].plot(history["round"], [r * 100 for r in history["recall"]], label='Recall (%)', color='#d62728', lw=1.5, ls='--')
    axes[1].set_title("FedAvg Global Model Convergence Across Rounds", fontsize=12)
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Metric Percentage (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    run_federated_training(num_clients=20, rounds=30, local_epochs=2, alpha=0.01)
