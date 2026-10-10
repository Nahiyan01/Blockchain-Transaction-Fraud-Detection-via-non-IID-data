"""
Model Definitions for Blockchain Fraud Detection
Based on: "Interpretable Federated Learning for Fraud Detection in Non-IID Blockchain Transactions"
Authors: Mahmudul Hasan et al.

Implements the paper's optimal architecture:
- 2-Layer Neural Network (2NN)
- Stochastic Gradient Descent (SGD) with momentum
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

def create_2nn_model(input_dim: int = 20, learning_rate: float = 0.02, optimizer_name: str = 'sgd'):
    """
    Constructs the 2-Layer Neural Network (2NN) described in the research paper.
    
    Architecture:
      Input (20 features)
      -> Dense(64, ReLU) + BatchNormalization + Dropout(0.2)
      -> Dense(32, ReLU) + Dropout(0.1)
      -> Output: Dense(1, Sigmoid)
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,), name="input_features"),
        
        # Layer 1
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name="dense_layer_1"),
        layers.BatchNormalization(name="batch_norm_1"),
        layers.Dropout(0.2, name="dropout_1"),
        
        # Layer 2
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name="dense_layer_2"),
        layers.Dropout(0.1, name="dropout_2"),
        
        # Output Layer (Binary classification: Fraud = 1, Normal = 0)
        layers.Dense(1, activation='sigmoid', name="fraud_output")
    ], name="Blockchain_Fraud_2NN")

    # Optimizer selection (Paper demonstrates SGD outperforming Adam under Non-IID skew)
    if optimizer_name.lower() == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer_name.lower() == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'adagrad':
        opt = optimizers.Adagrad(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

if __name__ == "__main__":
    test_model = create_2nn_model(input_dim=20, learning_rate=0.02, optimizer_name='sgd')
    test_model.summary()
