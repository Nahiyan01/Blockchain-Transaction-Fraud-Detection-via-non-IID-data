"""
Centralized Baseline Training Script
Paper: "Interpretable Federated Learning for Fraud Detection in Non-IID Blockchain Transactions"
Authors: Mahmudul Hasan et al.

Trains the 2-Layer Neural Network (2NN) with SGD optimizer on unified data
to establish the centralized upper-bound performance benchmark.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

from models import create_2nn_model

def run_centralized_training(data_dir="processed_data", output_dir="results/centralized"):
    os.makedirs(output_dir, exist_ok=True)
    print("="*70)
    print(" CENTRALIZED BASELINE TRAINING (2NN + SGD)")
    print("="*70)

    # 1. Load preprocessed datasets
    print("[1/5] Loading preprocessed numpy arrays...")
    X_train = np.load(os.path.join(data_dir, "X_train_scaled.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test_scaled.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    with open(os.path.join(data_dir, "dataset_summary.json"), "r") as f:
        meta = json.load(f)
    class_weights = {int(k): float(v) for k, v in meta["class_weights"].items()}

    print(f"      Train samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"      Test samples:  {X_test.shape[0]}")
    print(f"      Class weights: {class_weights}")

    # 2. Build model
    print("[2/5] Initializing 2-Layer Neural Network (2NN) with SGD optimizer...")
    model = create_2nn_model(input_dim=X_train.shape[1], learning_rate=0.03, optimizer_name='sgd')

    # 3. Train model
    print("[3/5] Training 2NN model (Epochs: 40, Batch size: 64)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=40,
        batch_size=64,
        class_weight=class_weights,
        verbose=1
    )

    # 4. Evaluate on Test Set
    print("[4/5] Evaluating performance on test set...")
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "-"*50)
    print(" TEST EVALUATION RESULTS:")
    print(f"   Accuracy:    {acc * 100:.2f}%")
    print(f"   Precision:   {prec * 100:.2f}%")
    print(f"   Recall:      {rec * 100:.2f}%")
    print(f"   F1-Score:    {f1 * 100:.2f}%")
    print(f"   ROC-AUC:     {roc_auc:.4f}")
    print("-"*50)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    # 5. Save Artifacts & Visualizations
    print("[5/5] Saving model weights and evaluation plots...")
    model_save_path = os.path.join(output_dir, "centralized_2nn_sgd.keras")
    model.save(model_save_path)

    metrics = {
        "model": "2NN",
        "optimizer": "SGD",
        "regime": "Centralized",
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist()
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot Confusion Matrix and Training History
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    axes[0].set_title(f"Centralized 2NN Confusion Matrix\n(F1-Score: {f1*100:.2f}%)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Training Loss Curve
    axes[1].plot(history.history['loss'], label='Train Loss', color='#1f77b4', lw=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', lw=2)
    axes[1].set_title("Training & Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Crossentropy Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "centralized_performance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"All centralized results saved to: {output_dir}")
    return metrics

if __name__ == "__main__":
    run_centralized_training()
