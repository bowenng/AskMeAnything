def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def create_look_ahead_mask(sequence_length):
    """
    Args:
        sequence_length: the length of input sequence
    Returns:
        look_ahead_mask: shape (sequence_length, sequence_length)
        
    e.g
    sequence_lenght = 3
    look_ahead_mask = [
        [0, 1, 1], on predicting the 1st word, only the 0th word (the <START/> token) will be used
        [0, 0, 1], on predicting the 2nd word, only the 1st word and the 0th word can be used
        [0, 0, 0]  and so on
    ]
    """
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((sequence_length, sequence_length)), -1, 0)
    return look_ahead_mask

def create_padding_mask(sparse_input_sequence):
    """
    Args:
        sparse_input_sequence: shape(batch_size, sequence_length) e.g [0, 1, 3, 0, 5, 13]
    Returns:
        padding_mask: boolean mask where 1s indicates padding. e.g [1, 0, 0, 1, 0, 0] for the example input
    """
    mask = tf.cast(tf.math.equal(sparse_input_sequence, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]
    