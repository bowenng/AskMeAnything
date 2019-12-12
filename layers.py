import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, d_model, n_layers, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.position_encoding = PositionalEncoding()
        self.encoder_layers = [EncoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, x, is_training, padding_mask):
        """
        Args: 
            position_encoded_embeddings: shape (batch size, sequence length, embedding dimension)
        """
        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.dropout(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, is_training, padding_mask)
        return x # x.shape == (batch_size, input_seq_len, d_model)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, feed_forward_d_ff, dropout_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)

        self.ffn = PointWiseFeedForwardDense(d_model, feed_forward_d_ff)

        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, x, is_training, padding_mask):
        attention = self.mha(x, x, x, padding_mask)
        attention = self.dropout1(attention, training=is_training)
        output1 = self.layernorm1(attention + x)
        
        ffoutput = self.ffn(output1)
        ffoutput = self.dropout2(ffoutput, training=is_training)
        output2 = self.layernorm2(ffoutput + output1)
        return output2

    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, d_model, n_layers, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(output_vocab_size, d_model)
        self.position_encoding = PositionalEncoding()
        self.decoder_layers = [DecoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, encoder_output, is_training, look_ahead_mask, padding_mask):
        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.dropout(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, is_training, look_ahead_mask, padding_mask)
        return x # x.shape == (batch_size, target_seq_len, d_model)
    
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, feed_forward_d_ff, dropout_rate=0.1):
        super().__init__()
        self.mha_decoder_decoder = MultiHeadAttention(d_model, n_heads)
        self.mha_encoder_decoder = MultiHeadAttention(d_model, n_heads)
        self.ffn = PointWiseFeedForwardDense(d_model, feed_forward_d_ff)

        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, encoder_output, is_training, look_ahead_mask, padding_mask):
        self_attention = self.mha_decoder_decoder(x, x, x, look_ahead_mask)
        self_attention = self.dropout1(self_attention, training=is_training)
        output1 = self.layernorm1(self_attention + x)
        
        encoder_decoder_attention = self.mha_encoder_decoder(output1, encoder_output, encoder_output, padding_mask)
        encoder_decoder_attention = self.dropout2(encoder_decoder_attention, training=is_training)
        output2 = self.layernorm2(encoder_decoder_attention + output1)
        
        ffoutput = self.ffn(output2)
        ffoutput = self.dropout3(ffoutput, training=is_training)
        output3 = self.layernorm3(ffoutput + output2)
        return output3
    
    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads):
        """
        Args: 
            d_model: the dimension of input/output of each sublayer in the encoder/decoder

            n_heads: number of heads to split
        Returns:
            aggregated_attention: aggregated multihead attention using a final dense layer
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # number of heads must be divisible by model dimension
        assert d_model % self.n_heads == 0
        
        self.depth = d_model // n_heads # depth of q, k, v
        

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.wout = tf.keras.layers.Dense(d_model)
        
        
    def call(self, query_sequence, key_sequence, value_sequence, mask=None):
        batch_size = tf.shape(query_sequence)[0]
        
        q = self.wq(query_sequence)
        k = self.wk(key_sequence)
        v = self.wv(value_sequence)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        attention = self.calculate_attention(q, k, v, mask) # (batch_size, n_heads, seq_len_q, depth)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) # (batch_size, seq_len_q, n_heads, depth)
        attention = tf.reshape(attention, [batch_size, -1, self.d_model])
        
        return self.wout(attention)
        
    def split_heads(self, x):
        """Split the last dimension into (n_heads, depth).
        Transpose the result such that the shape is (batch_size, n_heads, seq_len, depth)
        """
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth)) # (batch_size, seq_len, n_heads, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3]) # (batch_size, n_heads, seq_len, depth)
    
    def calculate_attention(self, q, k, v, mask):
        """
        assume depth = depth_q = depth_k = depth_v
        Args:
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_k, depth)
            v: value shape == (..., seq_len_v, depth)
            mask: Float tensor with shape broadcastable 
              to (..., seq_len_q, seq_len_k).

        Returns:
            attention: shape (..., seq_len_q, depth_v)
        """

        logit = tf.matmul(q, k, transpose_b=True) # Q matmul K_Tranpose
        sqrt_d = tf.math.sqrt(tf.cast(self.depth, tf.float32)) # sqrt(d)
        logit = tf.truediv(logit, sqrt_d) # QK_t/sqrt(d_v)
        if(mask is not None):
            logit += mask*1e-9 # make logits of words that shouldn't be used -inf, so after softmax they will be close to 0
        attention_weights = tf.nn.softmax(logit, axis=-1)
        attention = tf.matmul(attention_weights, v)
        return attention
    
    
class PointWiseFeedForwardDense(tf.keras.layers.Layer):
    def __init__(self, d_model , d_ff):
        """
        Args:
            d_model: dimension of the encoder/decoder
            d_ff: dimension of the intermediate layer
        """
        super().__init__()
        self.dense_hidden = tf.keras.layers.Dense(d_ff, activation='relu')
        self.dense_out = tf.keras.layers.Dense(d_model)
        
    def call(self, inputs):
        x = self.dense_hidden(inputs)
        x = self.dense_out(x)
        return x
    

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, embeddings):
        embedding_shape = tf.shape(embeddings)
        sequence_length = embedding_shape[-2]
        d_embedding = embedding_shape[-1]
        position_encoding = self.create_position_encoding(sequence_length, d_embedding)
        embeddings += position_encoding
        return embeddings
        
    def create_position_encoding(self, sequence_length, d_embedding):
        """
        Args:
            sequence_length: the length of input sequence. 
            d_embedding: the dimension of the embedding space. i.e 
        Returns:
            positional_encoding: shape (1, sequence_length, d_embedding)
            Note: the output is unsequeeze at dimension 1 to enable easier broadcasting
        """

        # position[i] = the i-th index the sequence
        positions = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
        # embedding_indices[i] = the i-th index in the embedding 
        # Note: embedding_indices has length d_embedding/2, because sin and cos share the same input
        # e.g [0, 2, 4, 6] for d_embedding = 8
        embedding_indices = tf.range(d_embedding, delta=2, dtype=tf.float32)[tf.newaxis:]

        # inner = pos/10000^(2i/EmbeddingDimension), i.e the input to sin and cos
        inner = positions / 10000**(embedding_indices/tf.cast(d_embedding, tf.float32))
        sin_position_encodings = tf.math.sin(inner)
        cos_position_encodings = tf.math.cos(inner)
        # to create alternating sin and cos encodings, we use a hack: we expand dim at the last axis and concatenate the resulting tensors
        sin_position_encodings = tf.expand_dims(sin_position_encodings, axis=-1)
        cos_position_encodings = tf.expand_dims(cos_position_encodings, axis=-1)
        position_encodings = tf.concat([sin_position_encodings, cos_position_encodings], axis=-1)
        position_encodings = tf.reshape(position_encodings, (1, sequence_length, d_embedding))
        return position_encodings