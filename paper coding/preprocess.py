"""
Preprocessing Pipeline for Ethereum Blockchain Fraud Detection
Based on the paper: "Interpretable Federated Learning for Fraud Detection in Non-IID Blockchain Transactions"
Authors: Mahmudul Hasan et al. (Blockchain: Research and Applications)

Stage 1: Preprocessing & Imputation
- Drops metadata/ID columns (Index, Address, Unnamed: 0)
- Detects and removes zero-variance features
- Imputes missing values:
    * Continuous/Numerical features -> Mean
    * Categorical features -> Mode
- Refines dataset into 20 core tabular features across 4 transaction domains:
    1. ERC20 Token Dynamics (Paper's #1 & #4 Indicators)
    2. Ether Balances (Paper's #2 Indicator)
    3. Smart Contract Interactions
    4. Temporal Intervals (Paper's #3 Indicator)
- Performs stratified Train-Test split (80/20)
- Applies feature scaling (StandardScaler) fitted on training data to prevent leakage
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

def load_and_clean_data(raw_csv_path: str):
    print(f"[1/6] Loading raw dataset from: {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    print(f"      Initial dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Clean whitespace from column headers
    df.columns = df.columns.str.strip()

    # 2. Drop administrative / ID columns
    id_cols = ['Unnamed: 0', 'Index', 'Address']
    existing_id_cols = [c for c in id_cols if c in df.columns]
    print(f"[2/6] Dropping non-predictive identifier columns: {existing_id_cols}")
    df = df.drop(columns=existing_id_cols)

    # 3. Detect and remove zero-variance columns
    num_cols_all = df.select_dtypes(include=[np.number]).columns
    zero_var_cols = [c for c in num_cols_all if c != 'FLAG' and df[c].var() == 0]
    print(f"[3/6] Identified {len(zero_var_cols)} zero-variance columns:")
    for col in zero_var_cols:
        print(f"      - {col} (variance = 0)")
    df = df.drop(columns=zero_var_cols)
    print(f"      Shape after dropping zero-variance columns: {df.shape}")

    # 4. Handle Missing Values
    print("[4/6] Handling missing values via paper specifications (Continuous -> Mean, Categorical -> Mode)...")
    num_cols = df.select_dtypes(include=[np.number]).columns.drop('FLAG', errors='ignore')
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Numeric imputation: Mean
    null_counts_num = df[num_cols].isnull().sum().sum()
    print(f"      Imputing {null_counts_num} missing values in numerical columns with Mean...")
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # Categorical imputation: Mode & label encoding
    for c in cat_cols:
        null_count_cat = df[c].isnull().sum()
        mode_val = df[c].mode()[0]
        print(f"      Imputing {null_count_cat} missing values in categorical column '{c}' with Mode ('{mode_val}')...")
        df[c] = df[c].fillna(mode_val)
        df[c] = df[c].astype('category').cat.codes

    # 5. Extract 20 Core Features
    print("[5/6] Refining to the 20 Core Tabular Features established in the research paper...")
    core_20_features = [
        # Domain 1: ERC20 Token Dynamics (10 features)
        'ERC20 uniq rec token name',       # Indicator #1: High token name diversity indicates phishing/malicious contracts
        'ERC20 max val sent',              # Indicator #4: Sudden value spike in ERC20 transfers
        'ERC20 avg val sent',              # Indicator #4: Monetary deviation from baseline
        'ERC20 min val sent',
        'ERC20 max val rec',
        'ERC20 avg val rec',
        'ERC20 min val rec',
        'Total ERC20 tnxs',
        'ERC20 uniq sent addr',
        'ERC20 uniq rec addr',

        # Domain 2: Ether Balances (6 features)
        'total ether balance',             # Indicator #2: Low or erratic balances correlate with disposable scam accounts
        'total Ether sent',
        'total ether received',
        'min val sent',
        'max val sent',
        'avg val sent',

        # Domain 3: Smart Contract Interactions (2 features)
        'Number of Created Contracts',
        'ERC20 uniq rec contract addr',

        # Domain 4: Temporal Intervals (2 features)
        'Avg min between received tnx',    # Indicator #3: Extremely short gaps reflect automated bot fraud
        'Time Diff between first and last (Mins)'
    ]

    # Verify all 20 features exist in dataframe
    missing_core = [f for f in core_20_features if f not in df.columns]
    if missing_core:
        raise ValueError(f"Missing core features in dataset: {missing_core}")

    df_20 = df[core_20_features + ['FLAG']].copy()
    print(f"      Successfully extracted 20 features + target 'FLAG'. Shape: {df_20.shape}")

    return df_20, core_20_features

def preprocess_and_split(df_20: pd.DataFrame, core_features: list, output_dir: str):
    print("[6/6] Splitting and normalizing data...")
    os.makedirs(output_dir, exist_ok=True)

    X = df_20[core_features].values
    y = df_20['FLAG'].values

    # Class balance statistics
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"      Class distribution: Normal (0): {class_dist[0]} ({counts[0]/len(y)*100:.2f}%), Fraud (1): {class_dist[1]} ({counts[1]/len(y)*100:.2f}%)")

    # Balanced class weights (for model loss calculation)
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = {int(k): float(v) for k, v in zip(np.unique(y), weights)}
    print(f"      Computed Balanced Class Weights: {class_weights}")

    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Standard Scaling (fitted strictly on train to avoid data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save artifacts
    print(f"      Saving cleaned datasets and preprocessing artifacts to '{output_dir}'...")

    # 1. Cleaned 20-features CSV (unscaled, human-readable)
    cleaned_csv_path = os.path.join(output_dir, "cleaned_ethereum_20features.csv")
    df_20.to_csv(cleaned_csv_path, index=False)

    # 2. Scaled numpy arrays
    np.save(os.path.join(output_dir, "X_train_scaled.npy"), X_train_scaled)
    np.save(os.path.join(output_dir, "X_test_scaled.npy"), X_test_scaled)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    # 3. Scaler object
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    # 4. Metadata summary JSON
    summary = {
        "dataset_name": "Ethereum Fraud Detection (Kaggle)",
        "total_samples": int(len(df_20)),
        "num_features": len(core_features),
        "features": core_features,
        "target": "FLAG",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "class_distribution": class_dist,
        "class_weights": class_weights,
        "scaler": "StandardScaler"
    }
    with open(os.path.join(output_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*70)
    print(" PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Artifacts saved:")
    print(f"  - Cleaned 20-feature dataset: {cleaned_csv_path}")
    print(f"  - Scaled train features:      {os.path.join(output_dir, 'X_train_scaled.npy')}")
    print(f"  - Scaled test features:       {os.path.join(output_dir, 'X_test_scaled.npy')}")
    print(f"  - Train labels:               {os.path.join(output_dir, 'y_train.npy')}")
    print(f"  - Test labels:                {os.path.join(output_dir, 'y_test.npy')}")
    print(f"  - Fitted Scaler:              {os.path.join(output_dir, 'scaler.joblib')}")
    print(f"  - Dataset Summary:            {os.path.join(output_dir, 'dataset_summary.json')}")
    print("="*70)

if __name__ == "__main__":
    # Auto-detect raw dataset path whether running from workspace root or paper coding directory
    candidates = [
        os.path.join("transaction_dataset.csv", "transaction_dataset.csv"),
        os.path.join("..", "transaction_dataset.csv", "transaction_dataset.csv"),
    ]
    raw_path = next((p for p in candidates if os.path.exists(p)), None)
    if not raw_path:
        raise FileNotFoundError("Could not find transaction_dataset.csv in current or parent directory.")

    output_directory = "processed_data"
    df_cleaned, core_feats = load_and_clean_data(raw_path)
    preprocess_and_split(df_cleaned, core_feats, output_directory)
