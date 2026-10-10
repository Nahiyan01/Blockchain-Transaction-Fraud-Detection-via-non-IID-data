"""
Stage 5: Post-Hoc Explainable AI (XAI) Script
Paper: "Interpretable Federated Learning for Fraud Detection in Non-IID Blockchain Transactions"
Authors: Mahmudul Hasan et al. (Blockchain: Research and Applications)

Implements:
- Central post-hoc SHAP analysis (Global feature attribution)
- LIME instance explanation (Local audit trail for flagged fraud transactions)
- Replicates the paper's core discovery: Top fraud indicators
  (#1: ERC20 Uniq Rec Token Name, #2: Total Ether Balance, #3: Avg Min Between Received Tnx, etc.)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
from lime import lime_tabular

def run_explainability(
    data_dir="processed_data",
    model_path="results/federated/federated_global_2nn_sgd.keras",
    output_dir="results/explainability"
):
    os.makedirs(output_dir, exist_ok=True)
    print("="*75)
    print(" STAGE 5: POST-HOC EXPLAINABLE AI (XAI) PIPELINE")
    print("="*75)

    # 1. Load Data & Metadata
    print("[1/4] Loading preprocessed test data & feature metadata...")
    X_train = np.load(os.path.join(data_dir, "X_train_scaled.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test_scaled.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    with open(os.path.join(data_dir, "dataset_summary.json"), "r") as f:
        meta = json.load(f)
    feature_names = meta["features"]

    # 2. Load Model
    print(f"[2/4] Loading global aggregated model from: {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Predict function wrapper
    def model_predict_proba(x):
        preds = model.predict(x, verbose=0).ravel()
        # Return probability of [Normal, Fraud]
        return np.column_stack([1.0 - preds, preds])

    # 3. Central Post-Hoc SHAP Analysis
    print("[3/4] Computing SHAP values (KernelExplainer on central global model)...")
    # Use 100 representative background samples (k-means summary or random sample)
    background_data = shap.kmeans(X_train, 25)
    explainer = shap.KernelExplainer(lambda x: model.predict(x, verbose=0).ravel(), background_data)

    # Explain 100 test instances
    test_sample = X_test[:100]
    shap_values = explainer.shap_values(test_sample)

    # Plot 1: SHAP Global Feature Importance Bar Plot
    plt.figure(figsize=(10, 8))
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    top_features = [feature_names[i] for i in sorted_idx[:15]]
    top_scores = mean_abs_shap[sorted_idx[:15]]

    plt.barh(range(len(top_features)), top_scores[::-1], color='#2b5c8f')
    plt.yticks(range(len(top_features)), top_features[::-1], fontsize=10)
    plt.xlabel("Mean |SHAP Value| (Impact on Model Fraud Prediction)", fontsize=11)
    plt.title("Post-Hoc Global Feature Importance (SHAP)", fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    shap_bar_path = os.path.join(output_dir, "shap_global_feature_importance.png")
    plt.savefig(shap_bar_path, dpi=300)
    plt.close()
    print(f"      Saved SHAP feature importance plot to: {shap_bar_path}")

    # Plot 2: SHAP Summary Beeswarm Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, test_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm Plot (Directional Feature Impact)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    shap_summary_path = os.path.join(output_dir, "shap_beeswarm_summary.png")
    plt.savefig(shap_summary_path, dpi=300)
    plt.close()
    print(f"      Saved SHAP summary beeswarm plot to: {shap_summary_path}")

    # 4. LIME Local Audit Explanation for Fraudulent Transactions
    print("[4/4] Generating LIME audit trail on confirmed fraudulent transaction...")
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["Normal", "Fraud"],
        mode="classification",
        random_state=42
    )

    # Find first true positive (actual fraud correctly predicted)
    preds_test = model.predict(X_test, verbose=0).ravel()
    fraud_indices = np.where((y_test == 1) & (preds_test >= 0.5))[0]
    sample_idx = fraud_indices[0] if len(fraud_indices) > 0 else 0

    exp = lime_explainer.explain_instance(
        data_row=X_test[sample_idx],
        predict_fn=model_predict_proba,
        num_features=10
    )

    lime_fig = exp.as_pyplot_figure()
    plt.title(f"LIME Local Explanation (Fraud Account Index #{sample_idx})\nPredicted Fraud Prob: {preds_test[sample_idx]*100:.1f}%", fontsize=11, fontweight='bold')
    plt.tight_layout()
    lime_plot_path = os.path.join(output_dir, "lime_local_fraud_audit.png")
    plt.savefig(lime_plot_path, dpi=300)
    plt.close()
    print(f"      Saved LIME local audit plot to: {lime_plot_path}")

    # Save XAI rankings to JSON
    xai_ranking = {
        feature_names[i]: float(mean_abs_shap[i]) for i in sorted_idx
    }
    with open(os.path.join(output_dir, "shap_feature_importance.json"), "w") as f:
        json.dump(xai_ranking, f, indent=4)

    print("\n" + "="*70)
    print(" STAGE 5 EXPLAINABILITY COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Top Fraud Indicators Discovered by SHAP:")
    for rank, (feat, score) in enumerate(list(xai_ranking.items())[:5], 1):
        print(f"   #{rank} {feat:<35s} | SHAP Impact: {score:.5f}")
    print("="*70)

if __name__ == "__main__":
    run_explainability()
