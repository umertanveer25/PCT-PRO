"""
Protocol-Aware Quantum Attention (PAQA)
---------------------------------------
A context-gating attention mechanism for clinical protocol compliance.
Authors: Yar Muhammad, Umer Tanveer (2025)
"""

import tensorflow as tf
from tensorflow.keras import layers

class ProtocolAwareQuantumAttention(layers.Layer):
    """
    PAQA enhances feature attention by injecting 'Protocol Constraints'.
    It uses quantum interference filtered by medical validity rules.
    """
    def __init__(self, d_model, n_heads=4, **kwargs):
        super(ProtocolAwareQuantumAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.vv = layers.Dense(d_model)
        
        # Protocol Constraint Head (Medical Validator)
        self.protocol_gate = layers.Dense(n_heads, activation='sigmoid')

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, protocol_features=None, mask=None):
        batch_size = tf.shape(q)[0]

        q_proj = self.wq(q)
        k_proj = self.wk(k)
        v_proj = self.vv(v)

        q_heads = self.split_heads(q_proj, batch_size)
        k_heads = self.split_heads(k_proj, batch_size)
        v_heads = self.split_heads(v_proj, batch_size)

        # Standard Attention Logic
        matmul_qk = tf.matmul(q_heads, k_heads, transpose_b=True)
        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Injection of Protocol Constraints
        if protocol_features is not None:
            # Generate a 'Validity Gate' for each head based on protocol context
            # (e.g. MQTT MsgType, QoS level, Packet Time Delta)
            validity = self.protocol_gate(protocol_features) # (Batch, n_heads)
            validity = tf.reshape(validity, (batch_size, self.n_heads, 1, 1))
            
            # Constrain attention where protocol rules are violated
            # Higher validity allows full attention; lower validity suppresses it
            scaled_attention_logits = scaled_attention_logits * validity

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v_heads)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))

        return concat_attention, attention_weights

if __name__ == "__main__":
    # Test PAQA
    paqa = ProtocolAwareQuantumAttention(d_model=32, n_heads=4)
    sample_input = tf.random.normal((5, 1, 32)) 
    sample_protocol = tf.random.normal((5, 10)) # 10 dummy protocol features
    
    out, weights = paqa(sample_input, sample_input, sample_input, protocol_features=sample_protocol)
    print(f"PAQA Output Shape: {out.shape}")
    print(f"Protocol-Gated Weights Sample: {weights[0, 0, 0, :5]}")
