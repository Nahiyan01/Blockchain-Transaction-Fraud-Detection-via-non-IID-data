# Interpretable Federated Learning for Blockchain Fraud Detection
## Comprehensive Research Guide, Methodology & Implementation Manual

> **Research Paper Reference:**  
> **Title:** *Interpretable Federated Learning for Fraud Detection in Non-IID Blockchain Transactions*  
> **Authors:** Mahmudul Hasan, Md Raihanul Islam Tomal, Md Palash Uddin  
> **Journal:** *Blockchain: Research and Applications* (Elsevier, 2026)  
> **DOI:** `10.1016/j.bcra.2026.100511`  
> **Dataset:** Ethereum Fraud Detection (Kaggle)

---

## 1. Executive Summary & Problem Formulation (For Your Teacher)

### 1.1 The Core Research Motivation
Decentralized blockchain systems like Ethereum offer transparent, pseudonymous transaction ledgers. However, this pseudonymity exposes networks to severe financial crimes:
- Phishing campaigns & fake token airdrops
- Illicit smart contract exploits & money laundering
- Automated high-frequency scam bots

Traditional Machine Learning (ML) requires **centralizing** all raw transaction logs onto a single server. In decentralized ecosystems, this approach breaks down due to three core bottlenecks:

1. **Privacy & Legal Violations (GDPR):** Decentralized cryptocurrency exchanges, financial institutions, and wallet providers cannot pool proprietary transaction histories without violating strict financial privacy regulations.
2. **Data Heterogeneity (Non-IID Drift):** In a decentralized network, different edge entities (e.g., retail individual wallets vs. institutional exchanges) hold vastly unequal transaction volumes and fraud ratios. This creates severe statistical drift (**Non-IID: Not Independent and Identically Distributed** data).
3. **Black-Box Opacity:** Financial regulatory authorities mandate human-understandable justifications before flagging an account or freezing assets.

### 1.2 The Paper's 5-Stage Framework
The authors introduce a unified **Federated Learning + Explainable AI (FL-XAI)** framework:
- **Stage 1 (Preprocessing & Imputation):** Cleans raw transaction logs, removes identifiers, strips zero-variance attributes, imputes missing values, and refines into **20 core domain features**.
- **Stage 2 (Client Partitioning):** Simulates **20 edge client nodes** under IID and Non-IID Dirichlet distributions ($\alpha \in \{1.0, 0.1, 0.01\}$) to rigorously test statistical heterogeneity.
- **Stage 3 (Local Model Training):** Compares lightweight **2-Layer Neural Networks (2NN)** vs. **Tabular Transformers (TabTF)** across three optimizers: **SGD**, **Adam**, and **AdaGrad**.
- **Stage 4 (FedAvg Aggregation):** Central server aggregates local model weights using Federated Averaging (**FedAvg**) over 200–500 communication rounds (with 50% partial or 100% full client participation).
- **Stage 5 (Post-Hoc XAI):** Central server executes **SHAP** (via Shapash) and **LIME** post-aggregation, adding **zero communication overhead** to edge clients.

---

## 2. Stage 1: Data Cleaning & Preprocessing (Completed)

### 2.1 Dataset Profile
- **Raw File:** `transaction_dataset.csv`
- **Initial Size:** 9,841 accounts $\times$ 51 features.
- **Target Variable (`FLAG`):**
  - `0` = Legitimate / Normal account (7,662 samples; **77.86%**)
  - `1` = Fraudulent account (2,179 samples; **22.14%**)
  - *Class Imbalance Ratio:* $\approx 3.5:1$

### 2.2 Preprocessing Pipeline (`preprocess.py`)
1. **Header Normalization:** Stripped accidental leading and trailing whitespace across all 51 column names.
2. **Dropping Identifiers:** Removed `['Unnamed: 0', 'Index', 'Address']`.
   > *Explanation for Teacher:* Wallet addresses are arbitrary hexadecimal identifiers. Retaining them causes tabular models to memorize addresses rather than learning transactional patterns, leading to severe data leakage.
3. **Stripping Zero-Variance Columns:** Dropped 7 columns that contain constant values across all 9,841 rows:
   - `ERC20 avg time between sent tnx`
   - `ERC20 avg time between rec tnx`
   - `ERC20 avg time between rec 2 tnx`
   - `ERC20 avg time between contract tnx`
   - `ERC20 min val sent contract`
   - `ERC20 max val sent contract`
   - `ERC20 avg val sent contract`
4. **Missing Value Imputation:**
   - **Continuous/Numeric features:** Imputed with **Mean** (13,264 null values imputed).
   - **Categorical features:** Imputed with **Mode** (3,568 null values imputed), followed by category encoding.
5. **Extraction of 20 Core Domain Features:**

| Dimension | Features Selected | Fraud Detection Domain Rationale |
| :--- | :--- | :--- |
| **1. ERC20 Token Dynamics** (10 features) | • `ERC20 uniq rec token name` *(Indicator #1)*<br>• `ERC20 max val sent` *(Indicator #4)*<br>• `ERC20 avg val sent` *(Indicator #4)*<br>• `ERC20 min val sent`<br>• `ERC20 max val rec`<br>• `ERC20 avg val rec`<br>• `ERC20 min val rec`<br>• `Total ERC20 tnxs`<br>• `ERC20 uniq sent addr`<br>• `ERC20 uniq rec addr` | Malicious accounts often interact with hundreds of random tokens (airdrop phishing scams) or execute sudden large token drainage. |
| **2. Ether Balances** (6 features) | • `total ether balance` *(Indicator #2)*<br>• `total Ether sent`<br>• `total ether received`<br>• `min val sent`<br>• `max val sent`<br>• `avg val sent` | Fraudulent wallets typically maintain near-zero balances—they operate as disposable "mule" nodes. |
| **3. Smart Contracts** (2 features) | • `Number of Created Contracts`<br>• `ERC20 uniq rec contract addr` | Scammers frequently deploy fresh phishing contracts or route illicit funds through mixer contracts. |
| **4. Temporal Intervals** (2 features) | • `Avg min between received tnx` *(Indicator #3)*<br>• `Time Diff between first and last (Mins)` | Scam bot scripts transact with sub-minute intervals, deviating sharply from human transaction habits. |

6. **Splitting & Scaling:**
   - Stratified 80/20 train/test split (7,872 training samples, 1,969 test samples).
   - `StandardScaler` ($\mu = 0, \sigma = 1$) fitted strictly on the training set.
   - Balanced class weights: $\text{Weight}(0) = 0.642, \text{Weight}(1) = 2.258$.

---

## 3. Stage 2: Client Partitioning (Federated Simulation)

In the real world, transactions are held across diverse wallets and exchanges. The paper simulates **$K = 20$ client nodes**.

### 3.1 Dirichlet Class-Label Allocation Formula
To model realistic statistical heterogeneity, data is allocated using a **Dirichlet distribution**:
$$p(x_1, \dots, x_k; \alpha_1, \dots, \alpha_k) = \frac{1}{\mathrm{B}(\boldsymbol{\alpha})} \prod_{i=1}^k x_i^{\alpha_i - 1}$$
Where the concentration parameter $\alpha$ controls data heterogeneity:
- **$\alpha = 1.0$ (Low Skew):** Mild heterogeneity across nodes.
- **$\alpha = 0.1$ (Moderate Skew):** Noticeable label and volume imbalance.
- **$\alpha = 0.01$ (Extreme Heterogeneity):** Clients hold nearly disjoint label subsets (e.g., some nodes hold almost exclusively legitimate transactions, while others hold fraud clusters).

---

## 4. Stage 3: Local Model Architecture & Optimizer Analysis

### 4.1 2-Layer Neural Network (2NN) Architecture
```
Input Layer:        20 features
Dense Layer 1:      64 units, ReLU activation, L2 Regularization (1e-4)
                    Batch Normalization, Dropout (0.2)
Dense Layer 2:      32 units, ReLU activation, L2 Regularization (1e-4)
                    Dropout (0.1)
Output Layer:       1 unit, Sigmoid activation (Binary Classification)
Total Parameters:   3,713 parameters (~14.5 KB)
```

### 4.2 Why SGD Outperforms Adam in Non-IID Regimes (Crucial Question for Your Teacher)
The paper's key finding is that under extreme Non-IID skew ($\alpha = 0.01$):
- **Why Adam fails:** Adaptive optimizers like Adam accumulate second-moment gradient statistics ($\nu_t$) locally. When data is non-IID, each client's gradients pull weights in radically opposing directions. This leads to catastrophic weight oscillations and "client drift" during central aggregation.
- **Why SGD succeeds:** Stochastic Gradient Descent (SGD) applies fixed-step, uniform updates that act as an **implicit natural regularizer**. It prevents any single skewed client from over-adapting, ensuring stable federated convergence.

### 4.3 Paper's Benchmark Comparison (High Heterogeneity $\alpha = 0.01$, 100% Participation)

| Model & Optimizer | Accuracy | F1-Score | Standard Deviation ($\sigma$) | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **2NN + SGD (Optimal)** | **97.11%** | **97.11%** | **$\pm 0.58\%$ (Rock solid)** | **210 s ($10\times$ faster)** |
| 2NN + Adam | 96.60% | 96.64% | $\pm 8.66\%$ (High variance) | 257 s |
| 2NN + AdaGrad | 65.21% | 68.10% | $\pm 6.52\%$ (Failed) | 292 s |
| TabTF + SGD | 94.06% | 93.72% | $\pm 9.41\%$ | 1,365 s |
| TabTF + Adam | 72.88% | 69.18% | $\pm 7.29\%$ (Catastrophic drift) | 2,473 s |

---

## 5. Stage 4: Federated Averaging (FedAvg) Protocol

### 5.1 Training Hyperparameters (from Section 4.1.1 of Paper)
- **Batch Size:** 64
- **Learning Rate:** 0.01
- **Local Epochs ($E$):** 2 local epochs per communication round
- **Communication Rounds ($T$):** 200 CRs for IID, 500 CRs for Non-IID
- **Client Participation ($C$):**
  - Partial Participation: $C = 0.5$ (50% of clients sampled randomly per round)
  - Full Participation: $C = 1.0$ (100% of clients participate every round)

### 5.2 FedAvg Aggregation Formula
At round $t$:
$$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k$$
Where $n_k$ is the number of samples held by client $k$, and $N = \sum_{k=1}^K n_k$.

---

## 6. Stage 5: Post-Hoc Explainability (XAI)

Post-aggregation, the central server runs:
- **SHAP (SHapley Additive exPlanations)** / **Shapash**: Computes global feature impact.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Generates instance-level audit trails for flagged accounts.
- **Advantage:** Centrally computed XAI incurs **zero communication or bandwidth overhead** for edge clients.

### Primary Fraud Predictors Discovered:
1. **`ERC20 uniq rec token name`**: #1 Indicator. Extreme token diversity indicates automated phishing smart contracts and airdrop scams.
2. **`total ether balance`**: #2 Indicator. Near-zero or erratic balance correlates with disposable money-mule accounts.
3. **`Avg min between received tnx`**: #3 Indicator. Rapid incoming bursts indicate automated bot networks.
4. **`ERC20 max val sent` & `avg val sent`**: #4 Indicator. Sudden monetary surges deviating from normal transaction history.

---

## 7. Directory Structure of the Implementation

```
Fraud Detection in Blockchain Transaction/
├── Block Chain Fraud Detection Flowchat .jpg
├── Fraud detection in BlockChain Transaction paper.pdf
├── transaction_dataset.csv/
│   └── transaction_dataset.csv
└── paper coding/
    ├── RESEARCH_METHODOLOGY_GUIDE.md    <-- This Guide
    ├── preprocess.py                    <-- Stage 1 Preprocessing Script
    ├── models.py                        <-- 2NN Model Architecture
    ├── train_centralized.py             <-- Centralized Baseline
    ├── train_federated.py               <-- Stage 2, 3, 4 Federated Simulation
    ├── processed_data/
    │   ├── cleaned_ethereum_20features.csv
    │   ├── X_train_scaled.npy
    │   ├── X_test_scaled.npy
    │   ├── y_train.npy
    │   ├── y_test.npy
    │   ├── scaler.joblib
    │   └── dataset_summary.json
    └── results/
        ├── centralized/
        │   ├── centralized_2nn_sgd.keras
        │   ├── centralized_performance.png
        │   └── metrics.json
        └── federated/
            ├── federated_global_2nn_sgd.keras
            ├── client_data_distribution.png
            ├── federated_convergence.png
            └── federated_metrics.json
```
