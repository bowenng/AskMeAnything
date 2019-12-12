import tensorflow as tf
from layers import Encoder, Decoder

class Transformer(tf.keras.Model):
    def __init__(self, input_vocab_size, output_vocab_size, d_model, n_layers, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(input_vocab_size, d_model, n_layers, n_heads, d_ff, dropout_rate)
        self.decoder = Decoder(output_vocab_size, d_model, n_layers, n_heads, d_ff, dropout_rate)
        self.final_output_dense = tf.keras.layers.Dense(output_vocab_size) # map decoder output from d_model to output_vocab_size
    
    def call(self, encoder_input_sequence, decoder_input_sequence, is_training=False, encoder_padding_mask=None, 
           look_ahead_mask=None, decoder_padding_mask=None):
        encoder_output = self.encoder(encoder_input_sequence, is_training, encoder_padding_mask)
        decoder_output = self.decoder(decoder_input_sequence, encoder_output, is_training, look_ahead_mask, decoder_padding_mask)
        output = self.final_output_dense(decoder_output)
        return output