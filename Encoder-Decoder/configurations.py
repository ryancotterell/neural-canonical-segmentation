def get_config_cs2en():
    config = {}

    # Settings ----------------------------------------------------------------
    config['identity_init'] = True
    config['all_identity_init'] = False
    config['early_stopping'] = False # this has no use for now
    
    # Model related -----------------------------------------------------------

    # Sequences longer than this will be discarded.
    config['seq_len'] = 50

    # Number of hidden units in encoder/decoder GRU.
    config['enc_nhids'] = 100
    config['dec_nhids'] = 100
    
    # For the initialization of the parameters.
    config['rng_value'] = 11

    # Dimension of the word embedding matrix in encoder/decoder.
    config['enc_embed'] = 300
    config['dec_embed'] = 300


    # Where to save the model.
    config['saveto'] = 'model'

    # Optimization related ----------------------------------------------------

    # Batch size.
    config['batch_size'] = 20

    # This many batches will be read ahead and sorted.
    config['sort_k_batches'] = 12

    # Optimization step rule.
    config['step_rule'] = 'AdaDelta'

    # Gradient clipping threshold.
    config['step_clipping'] = 1.

    # Std of weight initialization.
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers.
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers.
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout.
    config['dropout'] = 1.0

    # Vocabulary/dataset related ----------------------------------------------
    
    # Root directory for dataset.
    datadir = ''

    # Module name of the stream that will be used.
    config['stream'] = 'stream'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens.
    config['src_vocab_size'] = 159
    config['trg_vocab_size'] = 61

    # Special tokens and indexes.
    config['unk_id'] = 1
    config['bow_token'] = '<S>'
    config['eow_token'] = '</S>'
    config['unk_token'] = '<UNK>'

    # Print validation output to file.
    config['output_val_set'] = False

    # Validation output file.
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Beam-size.
    config['beam_size'] = 12

    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates.
    config['finish_after'] = 2

    # Reload model from files if exist.
    config['reload'] = True

    # Save model after this many updates.
    config['save_freq'] = 500

    # Show samples from model after this many updates.
    config['sampling_freq'] = 50

    # Show this many samples at each sampling.
    config['hook_samples'] = 2
    
    config['lang'] = None

    return config
