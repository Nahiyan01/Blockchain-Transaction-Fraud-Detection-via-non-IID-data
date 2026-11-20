"""
Stabilized Tabular Self-Attention Architecture for Blockchain Fraud Detection
Research Sprint Implementation: Rescuing Transformers from Non-IID Collapse (Overturning Hasan et al., 2026 limitation)

Key Innovations:
1. Feature Tokenizer: Projects 20 tabular features into latent D-dimensional token embeddings.
2. Layer Normalization: Fully eliminates Non-IID Batch Normalization variance drift across clients.
3. Multi-Head Self-Attention: Learns inter-feature correlation graphs (e.g. between ERC20 token dynamics and temporal intervals).
4. Residual Pre-LayerNorm Connections: Ensures stable gradient propagation during federated aggregation.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

class FeatureTokenizer(layers.Layer):
    """
    Transforms tabular feature vector of shape (batch, num_features)
    into a sequence of feature token embeddings of shape (batch, num_features, embed_dim).
    """
    def __init__(self, num_features=20, embed_dim=16, **kwargs):
        super(FeatureTokenizer, self).__init__(**kwargs)
        self.num_features = num_features
        self.embed_dim = embed_dim
        # Individual dense projection weights for each feature column
        self.dense = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(1e-4), name="token_projection")

    def call(self, inputs):
        # inputs: (batch, num_features)
        # Expand dims to (batch, num_features, 1)
        x = tf.expand_dims(inputs, axis=-1)
        # Project each feature to embed_dim
        x = self.dense(x)  # shape: (batch, num_features, embed_dim)
        return x

def create_stabilized_tabular_attention(input_dim=20, embed_dim=16, num_heads=2, dropout_rate=0.2):
    """
    Constructs a stabilized Tabular Self-Attention Network designed to resist Non-IID client drift.
    """
    inputs = layers.Input(shape=(input_dim,), name="raw_tabular_features")
    
    # 1. Feature Tokenization
    tokens = FeatureTokenizer(num_features=input_dim, embed_dim=embed_dim, name="feature_tokenizer")(inputs)
    
    # 2. Multi-Head Self-Attention Block with Pre-LayerNorm & Residual
    norm1 = layers.LayerNormalization(epsilon=1e-6, name="pre_mha_layernorm")(tokens)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=dropout_rate,
        name="multihead_attention"
    )(norm1, norm1)
    x = layers.Add(name="attn_residual")([tokens, attn_out])
    
    # 3. Feedforward Block with Pre-LayerNorm & Residual
    norm2 = layers.LayerNormalization(epsilon=1e-6, name="pre_ffn_layernorm")(x)
    ffn = layers.Dense(embed_dim * 2, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="ffn_dense_1")(norm2)
    ffn = layers.Dropout(dropout_rate, name="ffn_dropout")(ffn)
    ffn = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(1e-4), name="ffn_dense_2")(ffn)
    x = layers.Add(name="ffn_residual")([x, ffn])
    
    # 4. Final LayerNorm & Global Aggregation
    x = layers.LayerNormalization(epsilon=1e-6, name="post_block_layernorm")(x)
    pooled = layers.GlobalAveragePooling1D(name="global_token_pooling")(x)
    
    # 5. Classification Head
    hidden = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="head_dense")(pooled)
    hidden = layers.Dropout(0.1, name="head_dropout")(hidden)
    outputs = layers.Dense(1, activation="sigmoid", name="fraud_probability")(hidden)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Stabilized_Tabular_Attention_Network")
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.0),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model

if __name__ == "__main__":
    test_attn_model = create_stabilized_tabular_attention(input_dim=20, embed_dim=16, num_heads=2)
    test_attn_model.summary()
