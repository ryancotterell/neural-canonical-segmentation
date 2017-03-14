# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
An encoder-decoder model for canonical segmentation.
"""

from __future__ import print_function

from POET.filter_wrong_forms import Filter
import codecs, sys, re, getopt

import pickle as cPickle
import sys
import datetime
import time
import io

import codecs
import argparse
import logging
import pprint
import sys
import os

from MED import configurations as configurations
from MED.__init__ import main as rnnMain
from MED.__init__ import mainPredict as rnnPredict
from MED.stream import get_tr_stream, get_test_stream, get_dev_stream
import MED.sampling as sampling

###############################################################################
# IMPORTANT: Those variables should not be touched. Instead, use command line flags.

# Deciding on the use of POEM.
noFilter = True 
# Leave this False.
loadModel = False 
# Use trainRnn=True for training and trainRnn=False for evaluation.
trainRnn = True
# The index of the model in an ensemble to be trained; number of models in the ensemble for evaluation.
use_ensemble = 1
###############################################################################

logger = logging.getLogger(__name__)

def main(argv, resulting_predictions=None):
    global TASK

    ################################################  
    # Getting the command line arguments.
    options, remainder = getopt.gnu_getopt(sys.argv[1:], 'l:t:a:d:e:tr:te:p:c:s:', ['language=','task=', 'data=', 'ens=', 'train', 'test', 'poet', 'corpus', 'saveto='])
    PATH, TASK, DATAPATH, trainRnn, SAVETO, noFilter = './', 1, None, None, None, True
    for opt, arg in options:
        if opt in ('-l', '--language'):
            LANGUAGE = arg
        if opt in ('-t', '--task'):
            TASK = int(arg)
        if opt in ('-d', '--data'):
            DATAPATH = arg
        if opt in ('-e', '--ens'): 
            use_ensemble = int(arg)
        if opt in ('-tr', '--train'):
            trainRnn = True
        if opt in ('-te', '--test'):
            trainRnn = False
        if opt in ('-p', '--poet'):
            noFilter = False
        if opt in ('-s', '--saveto'):
            SAVETO = arg
      
    assert trainRnn != None 
    ################################################ 
   
    # POET, filtering words with no corresponding edit tree.
    if noFilter:
      answerFilter = None
    else:
      answerFilter = Filter(DATAPATH, LANGUAGE)
      
    middle = SAVETO
    if not noFilter:
      middle += 'POET/'
    if not os.path.exists('results/for_eval/' + middle):
      os.makedirs('results/for_eval/' + middle)

    result_for_eval = io.open('results/for_eval/' + middle + LANGUAGE + '-task' + str(TASK) + '-solution', 'w', encoding='utf-8')

    #testlines = [line.strip() for line in codecs.open('sample_data/SIGMORPHON/' + LANGUAGE + '-task' + str(TASK) + '-test-covered', "r", encoding="utf-8")] # USE THIS FOR REAL RESULTS!
    testlines = [line.strip() for line in codecs.open('sample_data/SIGMORPHON/' + LANGUAGE + '-task' + str(TASK) + '-dev', "r", encoding="utf-8")]
      
    if TASK == 1:
        # Initialize variables for error and filter analysis.
        errorCount = filterCount = correctFilter = rightCorrectionsCount = rightAnswersCount = totalCount = notFoundCount = 0
        for l in testlines:
            totalCount += 1
            split_result = l.split('\t')
            # Using a test file WITH solutions:
            if len(split_result) > 2:
              lemma, targetmsd, wordform = l.split('\t')
              for_eval_file = u'\t'.join(l.split('\t')[:-1])
            else:
	      lemma, targetmsd = l.split('\t')
	      wordform = None
	      for_eval_file = u'\t'.join(l.split('\t'))

            if (targetmsd, lemma) not in resulting_predictions:
              result_for_eval.write(for_eval_file + u'\t' + lemma + u'\n')
              errorCount += 1
              notFoundCount += 1
              logger.warning(u'NOT FOUND:\n' + targetmsd + lemma + '\n')
            guess = resulting_predictions[(targetmsd, lemma)][0]
            guess_prob = resulting_predictions[(targetmsd, lemma)][1]

            if not noFilter and not answerFilter.filterResult(lemma, targetmsd, guess):
              new_guess = answerFilter.correctResult(lemma, targetmsd, guess)
              if new_guess != guess:
                filterCount += 1
              guess = new_guess

	    if wordform != None:
	      if u'' + guess != u'' + wordform: # The original version. The other one is just to see how much is filtered out.
		errorCount += 1
	      else: 
		rightAnswersCount += 1
	    result_for_eval.write(for_eval_file + u'\t' + guess + u'\n')

        if wordform != None:
	  logger.info('Wrong answers: ' + str(errorCount) + '/' + str(len(testlines)) + '(' + str(notFoundCount) + ' not found)')
	  logger.info('Right answers: ' + str(rightAnswersCount) + '/' + str(len(testlines)))
	  logger.info('Accuracy: ' + str(rightAnswersCount*1.0/len(testlines)))
	  logger.info('Filtered: ' + str(filterCount))
 
    if TASK == 2:
        # Initialize variables for error and filter analysis.
        errorCount = filterCount = wrongFilter = correctFilter = rightCorrectionsCount = rightAnswersCount = totalCount = notFoundCount = corpusFilterRight = 0
        for l in testlines:
	    split_result = l.split('\t')
	    if len(split_result) > 3:
              sourcemsd, sourceform, targetmsd, targetform = l.split('\t')
              for_eval_file = u'\t'.join(l.split('\t')[:-1])
            else:
	      sourcemsd, sourceform, targetmsd = l.split('\t')
	      targetform = None
              for_eval_file = u'\t'.join(l.split('\t'))

            if (sourcemsd, targetmsd, sourceform) not in resulting_predictions:
              result_for_eval.write(for_eval_file + u'\t' + sourceform + u'\n')
              errorCount += 1
              notFoundCount += 1
              logger.warning(u'NOT FOUND:\n' + targetmsd + lemma + '\n')
            guess = resulting_predictions[(sourcemsd, targetmsd, sourceform)][0]
            old_guess = guess

            if targetform != None:
              if u'' + guess != u'' + targetform:
                errorCount += 1
                if u'' + old_guess == u'' + targetform:
		  wrongFilter += 1
              else: 
                rightAnswersCount += 1
                if u'' + old_guess != u'' + targetform:
		  correctFilter += 1
	    result_for_eval.write(for_eval_file + u'\t' + guess + u'\n')
              
        if targetform != None:
	  logger.info('Wrong answers: ' + str(errorCount) + '/' + str(len(testlines)) + '(' + str(notFoundCount) + ' not found)')
	  logger.info('Right answers: ' + str(rightAnswersCount) + '/' + str(len(testlines)))
	  logger.info('Accuracy: ' + str(rightAnswersCount*1.0/len(testlines)))
	  if not noFilter:
	    logger.info('Filtered: ' + str(filterCount))
	    logger.info('Filter errors: ' + str(wrongFilter))
	    logger.info('Corrected by filter: ' + str(correctFilter))
	else:
	  logger.info('Finished. Check results in ' + 'results/for_eval/' + middle + LANGUAGE + '-task' + str(TASK) + '-solution')

    if TASK == 3:
        errorCount = filterCount = correctFilter = rightCorrectionsCount = rightAnswersCount = totalCount = notFoundCount = 0
        for l in testlines:
	    split_result = l.split('\t')
	    if len(split_result) > 2: 
              sourceform, targetmsd, targetform = l.split('\t')
              for_eval_file = u'\t'.join(l.split('\t')[:-1])
            else:
	      sourceform, targetmsd = l.split('\t')
	      targetform = None
	      for_eval_file = u'\t'.join(l.split('\t'))

            if (targetmsd, sourceform) not in resulting_predictions:
              result_for_eval.write(for_eval_file + u'\t' + sourceform + u'\n')
              errorCount += 1
              notFoundCount += 1
              logger.warning(u'NOT FOUND:\n' + targetmsd + lemma + '\n')
            guess = resulting_predictions[(targetmsd, sourceform)][0]

            if targetform != None:
	      if u'' + guess != u'' + targetform:
		errorCount += 1
	      else: 
		rightAnswersCount += 1
            
            result_for_eval.write(for_eval_file + u'\t' + guess + u'\n')
            
        if targetform != None:
	  logger.info('Wrong answers: ' + str(errorCount) + '/' + str(len(testlines)) + '(' + str(notFoundCount) + ' not found)')
	  logger.info('Right answers: ' + str(rightAnswersCount) + '/' + str(len(testlines)))
	  logger.info('Accuracy: ' + str(rightAnswersCount*1.0/len(testlines)))
	  logger.info('Filtered: ' + str(filterCount))
	else:
	  logger.info('Finished. Check results in ' + 'results/for_eval/' + middle + LANGUAGE + '-task' + str(TASK) + '-solution')
            
     
# Prepares the RNN.
def prepareRnn(parser, lang, trainRnn, use_ensemble=0, data_path=None, SAVETO=None, the_task=None):

    # Getting arguments.
    if use_ensemble == 1 or trainRnn:
      parser.add_argument("--proto",  default="get_config_cs2en",
                        help="Prototype config to use for config")
      parser.add_argument("--bokeh",  default=False, action="store_true",
                        help="Use bokeh server for plotting")
      
    args = parser.parse_known_args()[0]

    # Getting model configurations.
    configuration = getattr(configurations, args.proto)()

    # Setting paths to source and target vocabularies.
    if the_task > 1:
      configuration['src_vocab'] = ['/mounts/Users/cisintern/kann/SIGMORPHON/Code/sigmorphon2016-master/data/forRnn/', '_src_voc_task' + str(the_task) + '.pkl']
      configuration['trg_vocab'] = ['/mounts/Users/cisintern/kann/SIGMORPHON/Code/sigmorphon2016-master/data/forRnn/', '_trg_voc_task' + str(the_task) + '.pkl']
    else:
      configuration['src_vocab'] = ['/mounts/Users/cisintern/kann/SIGMORPHON/Code/sigmorphon2016-master/data/forRnn/', '_src_voc.pkl']
      configuration['trg_vocab'] = ['/mounts/Users/cisintern/kann/SIGMORPHON/Code/sigmorphon2016-master/data/forRnn/', '_trg_voc.pkl']

    # Setting paths to source and target datasets.
    configuration['src_data'] = ['/mounts/Users/cisintern/kann/SIGMORPHON/Code/sigmorphon2016-master/data/forRnn/', '-task' + str(the_task) + '-train_src']
    configuration['trg_data'] = ['/mounts/Users/cisintern/kann/SIGMORPHON/Code/sigmorphon2016-master/data/forRnn/', '-task' + str(the_task) + '-train_trg']

    configuration['saveto'] += '_' + lang + '_' + str(configuration['enc_nhids']) + '_' + str(configuration['dec_nhids'])
    
    # Use the path to data (if given):
    assert data_path != None
    dataPath = data_path
   
    model_type_name = SAVETO
    configuration['saveto'] = model_type_name + 'task' + str(the_task) + '/' + 'Ens_' + str(use_ensemble) + '_'  + configuration['saveto']
      
    if trainRnn == False and 'track2' in configuration['saveto'] and 'task1' in configuration['saveto']:
      configuration['saveto'] = configuration['saveto'].split('task1')
      configuration['saveto'] = configuration['saveto'][0] + 'task2' + configuration['saveto'][1]
    
    if not os.path.exists(configuration['saveto']):
      os.makedirs(configuration['saveto'])

    if the_task > 1:
      no_char_file = open(dataPath + lang + '_number_chars_task' + str(the_task), 'rb')
    else:
      no_char_file = open(dataPath + lang + '_number_chars', 'rb')
    configuration['src_vocab_size'] = cPickle.load(no_char_file)
    configuration['trg_vocab_size'] = cPickle.load(no_char_file)
    no_char_file.close()

    if trainRnn:
      configuration['lang'] = lang
      configuration['src_data'] = data_path + lang + configuration['src_data'][1]
      configuration['trg_data'] = data_path + lang + configuration['trg_data'][1] 

      configuration['val_set'] = data_path + lang + '-task' + str(the_task) + '-dev_src'
      configuration['val_set_grndtruth'] = data_path + lang + '-task' + str(the_task) + '-dev_trg'
      copyConfig(configuration)
    else:
      configuration['src_data'] = data_path + lang + '-task' + str(the_task) + '-test_src'
      testlines = [line.strip() for line in codecs.open(configuration['src_data'], "r", encoding="utf-8")]
      configuration['batch_size'] = len(testlines) + 1

    configuration['src_vocab'] = data_path + lang + configuration['src_vocab'][1]
    configuration['trg_vocab'] = data_path + lang + configuration['trg_vocab'][1]
    
    logger.info("Model options:\n{}".format(pprint.pformat(configuration)))

    if trainRnn:
      rnnMain(configuration, use_ensemble, get_tr_stream(**configuration), get_dev_stream(config=configuration), args.bokeh)
    else:
      rnnPredict(configuration, get_test_stream(**configuration), use_ensemble, lang, the_task, args.bokeh) 

def convert_format(inp, task, track=1):
  split_word = inp.split(' ')
  if task == 1 and track == 1:
    orig_form_array = []
    trg_tag_array = []
    for sth in split_word:
      if u'=' in sth:
	trg_tag_array.append(sth)
      else:
	orig_form_array.append(sth)

    return (u','.join(trg_tag_array), u''.join(orig_form_array))
  
  if task == 1 and track == 2:
    orig_form_array = []
    trg_tag_array = []
    for sth in split_word:
      if u'OUT=' in sth:
	trg_tag_array.append(sth.split(u'UT=')[1])
      else:
	if not u'IN=LEMMA' in sth:
	  orig_form_array.append(sth)

    return (u','.join(trg_tag_array), u''.join(orig_form_array))
  
  if task == 2:
    orig_form_array = []
    orig_tag_array = []
    trg_tag_array = []
    for sth in split_word:
      if 'IN=' in sth:
	orig_tag_array.append(sth.split('IN=')[1])
      elif 'OUT=' in sth:
	trg_tag_array.append(sth.split('OUT=')[1])
      else:
	orig_form_array.append(sth)
    return (u','.join(orig_tag_array), u','.join(trg_tag_array), u''.join(orig_form_array))
  
  if task == 3:
    orig_form_array = []
    trg_tag_array = []
    for sth in split_word:
      if u'=' in sth:
	trg_tag_array.append(sth)
      else:
	orig_form_array.append(sth)

    return (u','.join(trg_tag_array), u''.join(orig_form_array))
  
  
# @pred_list: results of the single networks in a dictionary key -> (word, prob)
def ensemble_results(pred_list, method = 'adding_prob', task=2):
  total_preds = {}
  all_ensemble_results = {}
  
  # Define the track for task 1, because of the format.
  track = 1
  if task == 1:
    for old_key, value in pred_list[0].iteritems():
      if u'IN=LEMMA' in old_key:
	track = 2
	continue
      
  # adding_prob means that the single probs are added and then the highest one is chosen.
  if method == 'adding_prob':
    for resulting_predictions in pred_list:
        for key, value in resulting_predictions.iteritems():
	    orig_form, orig_tag, trg_tag = convert_format(key, task)
            if key not in total_preds:
	      total_preds[key] = {value[0]: value[1]}
	      all_ensemble_results[key] = set()
	    else:
	      if value[0] not in total_preds[key]:
		total_preds[key][value[0]] = 0
	      total_preds[key][value[0]] += value[1]
	      all_ensemble_results[key].add(value[0])
	      
  # adding_appearances means that the solution which was selected most is chosen.
  if method == 'adding_appearances':
    for resulting_predictions in pred_list:
        for old_key, value in resulting_predictions.iteritems():
	    key = convert_format(old_key, task, track)
            if key not in total_preds:
	      total_preds[key] = {value: 1}
	      all_ensemble_results[key] = set()
	    else:
	      if value not in total_preds[key]:
		total_preds[key][value] = 0
	      total_preds[key][value] += 1
	      all_ensemble_results[key].add(value)
	            
  # Now: get the maximum (=best answer).
  for key, value in total_preds.iteritems():
    for w in sorted(value, key=value.get, reverse=True):
      final_predictions[key] = (w, value[w]) # Add only the most frequent result.
      break
      
  return final_predictions, all_ensemble_results

def copyConfig(config):
  outfile = open(config['saveto']+ '/config', 'a')
  outfile.write('\n' + str(time.time()) + '\n')
  
  for k, v in config.iteritems():
    outfile.write(k + '\t' + str(v) + '\n')
  
  
if __name__ == "__main__":
    
    # Needed for redirection of the output.
    sys.stdout=codecs.getwriter('utf-8')(sys.stdout)
    
    # Getting the command line arguments.
    options, remainder = getopt.gnu_getopt(sys.argv[1:], 'l:t:a:d:e:tr:te:p:c:s:', ['language=','task=', 'data=', 'ens=', 'train', 'test', 'poet', 'corpus', 'saveto='])
    PATH, TASK, DATAPATH, SAVETO = './', 1, None, None
    for opt, arg in options:
        if opt in ('-l', '--language'):
            LANGUAGE = arg
        if opt in ('-t', '--task'):
            TASK = int(arg)
        if opt in ('-d', '--data'):
            DATAPATH = arg
        if opt in ('-e', '--ens'):
            use_ensemble = int(arg)
        if opt in ('-tr', '--train'):
            trainRnn = True
        if opt in ('-te', '--test'):
            trainRnn = False
        if opt in ('-p', '--poet'):
            if TASK == 1:
              logger.info('Using POET.')
            else:
              logger.error('POET cannot be used with any other task than task 1.')
              exit()
            noFilter = False
        if opt in ('-s', '--saveto'):
            SAVETO = arg
      
    assert (DATAPATH != None or not trainRnn) and trainRnn != None and (SAVETO != None or not trainRnn)
      
    ################################################ 
      
    parser = argparse.ArgumentParser()   
    the_new_path = 'results/' + SAVETO + 'task' + str(TASK) + '/' + LANGUAGE + '/'

    if trainRnn:
      prepareRnn(parser, LANGUAGE, trainRnn, use_ensemble, data_path=DATAPATH, SAVETO=SAVETO, the_task=TASK)
    else:
      if not os.path.exists(the_new_path):
        os.makedirs(the_new_path)
          
        for i in range(1, use_ensemble+1):
          prepareRnn(parser, LANGUAGE, trainRnn, i, data_path=DATAPATH, SAVETO=SAVETO, the_task=TASK)
      else:
	logger.info('\nThere are already results for this language. Using those...\n')
 
      # For predictions:
      final_predictions = {}
      collected_predictions = []
      all_ensemble_results = {}
   
      for i in range(1, use_ensemble+1):
        resulting_predictions = cPickle.load(open(the_new_path + 'Ens_' + str(i) + '_intermediate_results.pkl', 'rb'))
	collected_predictions.append(resulting_predictions)
      final_predictions, all_ensemble_results = ensemble_results(collected_predictions, 'adding_appearances', TASK)
      main(sys.argv, final_predictions)

    logger.info('INFO: Finished')
    
