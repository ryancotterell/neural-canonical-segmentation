# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function

import io
import logging
import numpy
import operator
import os
import sys
import re
import signal
import theano
import time
from time import gmtime, strftime

import cPickle

from blocks.extensions import TrainingExtension, SimpleExtension
from blocks.search import BeamSearch
from blocks.extensions.monitoring import MonitoringExtension
from blocks.monitoring.evaluators import AggregationBuffer, DatasetEvaluator
from stream import _ensure_special_tokens

from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)

class SamplingBase(object):
    """Utility class for BleuValidator and Sampler."""

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, vocab):
        try:
            return seq.tolist().index(vocab['</S>']) + 1
        except ValueError:
            return len(seq)

    def _oov_to_unk(self, seq, vocab_size, unk_idx):
        return [x if x < vocab_size else unk_idx for x in seq]

    def _idx_to_word(self, seq, ivocab):
        return u" ".join([ivocab.get(idx, u"<UNK>") for idx in seq])

class Sampler(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self, model, data_stream, hook_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, src_vocab_size=None, **kwargs):
                 #trg_ivocab=None, src_vocab_size=None, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.model = model
        self.hook_samples = hook_samples
        self.data_stream = data_stream
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_vocab_size = src_vocab_size
        self.is_synced = False
        self.sampling_fn = model.get_theano_function()

    def do(self, which_callback, *args):

        # Get dictionaries, this may not be the practical way.
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary.
        # WARNING: Source and target indices from data stream
        #  can be different.
        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)

        # Randomly select source samples from the current batch.
        # WARNING: Source and target indices from data stream
        #  can be different.
        batch = args[0]
        batch_size = batch['source'].shape[0]
        hook_samples = min(batch_size, self.hook_samples)

        sample_idx = numpy.random.choice(
            batch_size, hook_samples, replace=False)
        src_batch = batch[self.main_loop.data_stream.mask_sources[0]]
        
        trg_batch = batch[self.main_loop.data_stream.mask_sources[1]]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        for i in range(hook_samples):
            input_length = self._get_true_length(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)

            inp = input_[i, :input_length]
            
            _1, outputs, _2, _3, costs, _4 = (self.sampling_fn(inp[None, :]))
            outputs = outputs.flatten()
            costs = costs.T

            sample_length = self._get_true_length(outputs, self.trg_vocab)

            print(u"Input : ", self._idx_to_word(input_[i][:input_length],
                                                self.src_ivocab))
            print("Input in indices: ", input_[i][:input_length])
            print("Target: ", self._idx_to_word(target_[i][:target_length],
                                                self.trg_ivocab))
            print("Sample: ", self._idx_to_word(outputs[:sample_length],
                                                self.trg_ivocab))
            print("Sample in indices: ", outputs[:sample_length])
            print("Sample cost: ", costs[:sample_length].sum())
            print()
    
class AccuracyValidator(SimpleExtension, SamplingBase):
    """Stores the best model based on accuracy on the dev set."""

    def __init__(self, source_sentence, samples, model, data_stream,
                 config, n_best=1, track_n_models=1, trg_ivocab=None, **kwargs):
        super(AccuracyValidator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.best_accuracy = 0.0
        self.validation_results_file = self.config['saveto'] + '/validation/accuracies_' + self.config['lang']

        # Helpers
        self.vocab = None
        self.src_vocab = None
        self.src_ivocab = None
        self.trg_vocab = None
        self.trg_ivocab = None

        # The next two are hacks.
        self.unk_sym = '<UNK>'
        self.eos_sym = '</S>'
        self.unk_idx = None
        self.eos_idx = None
        self.sampling_fn = self.model.get_theano_function()
        
        self.eow_idx = 2
       
        # Create saving directory if it does not exist.
        validation_path = self.config['saveto'] + '/validation/'
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)
 
        res_file = open(self.validation_results_file, 'a')
        res_file.write(str(time.time()) + '\n')
        res_file.close()

    def do(self, which_callback, *args):
        # Evaluate and save if necessary.
        no_epochs_done = self.main_loop.status['epochs_done']
        self._evaluate_model(no_epochs_done)

    def _evaluate_model(self, no_epochs_done):
        logger.info("Started Validation.")
        val_start_time = time.time()
        error_count = 0
        total_count = 0

        # Get target vocabulary.
        if not self.trg_ivocab:
            # Load dictionaries and ensure special tokens exist.
            self.src_vocab = _ensure_special_tokens(
                cPickle.load(open(self.config['src_vocab'])),
                bos_idx=0, eos_idx=2, unk_idx=1)
            self.trg_vocab = _ensure_special_tokens(
                cPickle.load(open(self.config['trg_vocab'])),
                bos_idx=0, eos_idx=2, unk_idx=1)
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
            self.unk_idx = self.src_vocab[self.unk_sym]
            self.eos_idx = self.src_vocab[self.eos_sym]
      
        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file.
            """
            seq = line[0]
            input_ = numpy.tile(seq, (1, 1))
            
            seq2 = line[2]
            target_ = numpy.tile(seq2, (1, 1))

            batch_size = input_.shape[0]

            for j in range(batch_size):
        
              input_length = get_true_length(input_[j], self.src_vocab)
              target_length = get_true_length(target_[j], self.trg_vocab)
        
              inp = input_[j, :input_length]
              _1, outputs, _2, _3, costs, _4 = (self.sampling_fn(inp[None, :]))
              outputs = outputs.flatten()
              sample_length = get_true_length(outputs, self.trg_vocab)
        
              input_word = _idx_to_word(input_[j][:input_length], self.src_ivocab)
              target_word = _idx_to_word(target_[j][:target_length], self.trg_ivocab)
              predicted_word = _idx_to_word(outputs[:sample_length], self.trg_ivocab)
              if target_word != predicted_word:
                error_count += 1
              total_count += 1
        
        new_accuracy = (total_count - error_count)*1.0 / total_count
        self._save_model(new_accuracy)

        res_file = open(self.validation_results_file, 'a')
        res_file.write(str(no_epochs_done) + '\t' + str(new_accuracy) + '\t' + str(error_count) + '\t' + str(total_count) + '\n')
        res_file.close()
        logger.info("Validation finished. Current accuracy on dev set: " + str(new_accuracy))

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
           key=operator.attrgetter('bleu_score')).bleu_score < bleu_score:
            return True
        return False

    def _save_model(self, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print('New highscore ' + str(accuracy) + '!\nSaving the model... \n')

            model = ModelInfo(accuracy, self.config['saveto'])
            
            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))
             
            param_values = self.main_loop.model.get_parameter_values()
            param_values = {name.replace("/", "-"): param
                        for name, param in param_values.items()}
            numpy.savez(model.path, **param_values)
            with open(self.config['saveto'] + '/best_params.lg', 'a') as log_file:
              log_file.write(strftime("%Y-%m-%d %H:%M:%S"))
              log_file.write('\nBest params stored with validation score of ' + str(self.best_accuracy) + '\n')
            signal.signal(signal.SIGINT, s)           

class ModelInfo:
    """Utility class to keep track of evaluated models."""

    def __init__(self, bleu_score, path=None):
        self.bleu_score = bleu_score
        self.path = self._generate_path(path)

    def _generate_path(self, path):
        gen_path = os.path.join(
            path, 'best_params.npz')
        return gen_path
      
def get_true_length(seq, vocab):
    try:
        return seq.tolist().index(vocab['</S>']) + 1
    except ValueError:
        return len(seq)
            
def _idx_to_word(seq, ivocab):
    return u" ".join([ivocab.get(idx, u"<UNK>") for idx in seq])
