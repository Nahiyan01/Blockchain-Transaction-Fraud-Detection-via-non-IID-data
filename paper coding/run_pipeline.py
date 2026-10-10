"""
Master Execution & Presentation Pipeline
Blockchain Fraud Detection via Interpretable Federated Learning

Run this script to reproduce the entire 5-stage research workflow
or demonstrate the project results directly to your teacher.
"""

import os
import sys
import json
import time

def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def main():
    # Ensure current directory is the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print_header("INTERPRETABLE FEDERATED LEARNING FOR BLOCKCHAIN FRAUD DETECTION\n  Research Replication & Experimental Demonstration")
    print("  Reference Paper: Mahmudul Hasan et al., Blockchain: Research & Applications (2026)")
    print("  Dataset: Kaggle Ethereum Fraud Detection (9,841 accounts, 51 raw attributes)")
    print("  Optimal Configuration: 2-Layer Neural Network (2NN) + SGD Optimizer\n")

    print("Options:")
    print("  [1] Run Stage 1: Data Preprocessing & 20 Core Feature Extraction")
    print("  [2] Run Centralized Baseline: Train 2NN + SGD on Unified Data")
    print("  [3] Run Stages 2, 3 & 4: Federated Learning Simulation (20 Clients, Dirichlet alpha=0.01)")
    print("  [4] Run Stage 5: Post-Hoc Explainability (SHAP & LIME Plots)")
    print("  [5] Run Entire Pipeline (Stages 1 through 5)")
    print("  [6] Display Current Research Results & Comparison Table for Teacher")
    print("  [0] Exit")

    choice = input("\nEnter choice [0-6] (default=6 to view results): ").strip() or "6"

    if choice == "1":
        print_header("RUNNING STAGE 1: PREPROCESSING")
        os.system(f'"{sys.executable}" preprocess.py')

    elif choice == "2":
        print_header("RUNNING CENTRALIZED BASELINE (2NN + SGD)")
        os.system(f'"{sys.executable}" train_centralized.py')

    elif choice == "3":
        print_header("RUNNING STAGES 2, 3 & 4: FEDERATED LEARNING SIMULATION")
        os.system(f'"{sys.executable}" train_federated.py')

    elif choice == "4":
        print_header("RUNNING STAGE 5: POST-HOC EXPLAINABILITY (SHAP & LIME)")
        os.system(f'"{sys.executable}" explainability.py')

    elif choice == "5":
        print_header("RUNNING FULL PIPELINE (STAGES 1 - 5)")
        print("\n--> [Step 1/4] Preprocessing...")
        os.system(f'"{sys.executable}" preprocess.py')
        print("\n--> [Step 2/4] Centralized Training...")
        os.system(f'"{sys.executable}" train_centralized.py')
        print("\n--> [Step 3/4] Federated Simulation...")
        os.system(f'"{sys.executable}" train_federated.py')
        print("\n--> [Step 4/4] Explainability (SHAP & LIME)...")
        os.system(f'"{sys.executable}" explainability.py')
        display_results()

    elif choice == "6":
        display_results()

    else:
        print("Exiting.")

def display_results():
    print_header("EXPERIMENTAL RESULTS COMPARISON & PRESENTATION SUMMARY")

    # Load centralized metrics
    cent_path = os.path.join("results", "centralized", "metrics.json")
    fed_path = os.path.join("results", "federated", "federated_metrics.json")
    xai_path = os.path.join("results", "explainability", "shap_feature_importance.json")

    has_cent = os.path.exists(cent_path)
    has_fed = os.path.exists(fed_path)
    has_xai = os.path.exists(xai_path)

    print("\n1. PERFORMANCE BENCHMARK COMPARISON TABLE:\n")
    print(f"{'Regime / Model':<35} | {'Accuracy':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 85)

    # Paper's published figures
    print(f"{'Paper: 2NN + SGD (Non-IID a=0.01)':<35} | {'97.11%':<10} | {'97.11%':<10} | {'97.11%':<10} | {'--':<10}")
    print(f"{'Paper: TabTF + Adam (Non-IID a=0.01)':<35} | {'72.88%':<10} | {'72.88%':<10} | {'72.78%':<10} | {'--':<10}")

    if has_cent:
        with open(cent_path, "r") as f:
            c = json.load(f)
        print(f"{'Our Centralized Baseline (2NN + SGD)':<35} | {c['accuracy']*100:6.2f}%    | {c['recall']*100:6.2f}%    | {c['f1_score']*100:6.2f}%    | {c['roc_auc']:6.4f}")

    if has_fed:
        with open(fed_path, "r") as f:
            fl = json.load(f)
        print(f"{'Our Federated Model (20 Clients)':<35} | {fl['final_accuracy']*100:6.2f}%    | {fl['final_recall']*100:6.2f}%    | {fl['final_f1_score']*100:6.2f}%    | {fl['final_roc_auc']:6.4f}")

    print("\n" + "="*80)
    print("2. TOP 5 FRAUD PREDICTORS DISCOVERED (CENTRAL POST-HOC SHAP):")
    if has_xai:
        with open(xai_path, "r") as f:
            x = json.load(f)
        for rank, (feat, score) in enumerate(list(x.items())[:5], 1):
            print(f"   #{rank} {feat:<35} | Impact (SHAP value): {score:.5f}")
    else:
        print("   (Run Stage 5 to view SHAP rankings)")

    print("\n" + "="*80)
    print("3. GENERATED PLOTS & ARTIFACTS FOR PRESENTATION:")
    print("   * Data Partitioning Skew (Dirichlet): results/federated/client_data_distribution.png")
    print("   * Federated FedAvg Convergence Curve:  results/federated/federated_convergence.png")
    print("   * Centralized Baseline Performance:    results/centralized/centralized_performance.png")
    print("   * SHAP Global Feature Importance:      results/explainability/shap_global_feature_importance.png")
    print("   * SHAP Beeswarm Distribution Plot:     results/explainability/shap_beeswarm_summary.png")
    print("   * LIME Single Account Audit Trail:     results/explainability/lime_local_fraud_audit.png")
    print("   * Research Guide & Theory Manual:      RESEARCH_METHODOLOGY_GUIDE.md")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
