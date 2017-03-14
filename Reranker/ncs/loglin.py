from __future__ import division
import cPickle as pickle
from arsenal.alphabet import Alphabet
import enchant
from collections import defaultdict as dd
from random import shuffle

import logging
from logging.handlers import RotatingFileHandler
import sys

# numerical 
import editdistance, io
import numpy as np
from numpy import zeros, ones, eye
import theano
import theano.tensor as T

CORRECT = "CORRECT"

class LogisticRegression(object):
    """ Log-linear model """

    # N=threshold (5 in this case)
    def __init__(self, N, M):
        self.W = theano.shared(value=zeros((N), dtype=theano.config.floatX), name='W')
        self.U1 = theano.shared(value=eye(N), name='U1')

        self.b = theano.shared(value=zeros((N), dtype=theano.config.floatX), name='b')

        # input matrix of features (each row is a feature vector for the ith class)
        self.x = T.matrix('x', dtype='float64')
        # importance weights
        self.w = T.vector('w', dtype='float64')
        # ith 
        self.i = T.scalar('i', dtype='int64')
        self.gamma = T.scalar('gamma', dtype='float64')
        self.eta = T.scalar('eta', dtype='float64')

        # x is very sparse, so this is slow.
        self.logprob = (T.nnet.softmax(T.dot(T.nnet.sigmoid(T.dot(self.x, self.U1)), self.W)))
        self.cost = T.log(self.w * self.logprob[0])[self.i] - self.gamma * T.dot(self.W, self.W)
        self.decoded = T.log(self.w * self.logprob[0])
        self.f = theano.function(inputs=[self.x, self.w, self.i, self.gamma], outputs=self.cost)
        self.decode = theano.function(inputs=[self.x, self.w], outputs=self.decoded.argmax())
        
        self.parameters = [self.W, self.U1]
        self.gradients = T.grad(self.cost, self.parameters)

        self.step = theano.function(inputs=[self.x, self.w, self.i, self.gamma, self.eta], outputs=self.cost, \
                                    updates=[(p, p + self.eta * g) for (p, g) in zip(self.parameters, self.gradients)])


def f1_score(guess, gold):
    counter = 0
    for morpheme in guess:
      if morpheme in gold:
        counter += 1
    precision = (counter*1.0) / len(guess)
    
    counter = 0
    for morpheme in gold:
      if morpheme in guess:
        counter += 1
    recall = (counter*1.0) / len(gold)
    
    if precision == 0.0 or recall == 0.0:
      return 0.0
    return 2*precision*recall/(precision + recall)
     

class Model(object):
    """ log linear model """

    def __init__(self, lang, data_train, data_dev, train_size=8000):
        self.d = enchant.Dict(lang)
        self.data_train = data_train
        self.data_dev = data_dev
        self.train_size = train_size
        self.data = {}
        for k, v in self.data_dev.items():
            if k not in self.data:
                self.data[k] = v

        for i, (k, v) in enumerate(self.data_train.items()):
            if i == self.train_size:
                break
            if k not in self.data:
                self.data[k] = v
        self.model = None

    def vectorize(self, segmentations, maxn=200, threshold=5):
        """ vectorize to features for theano """

        lookup = {'11': 0, '22' : 1}
        index = Alphabet()
        count = dd(int)
        for segmentation in segmentations:
            for segment in segmentation:
                if segment not in lookup:
                    lookup[segment] = len(lookup)
                count[segment] += 1
        # create vectors
        self.N = threshold
        for k, v in count.items():
            if v > threshold:
                index.add(k)
                self.N += 1
        seg2vec = {}
        for seg, i in lookup.items():
            if i < 2:
                continue
            vec = zeros((self.N))
            if (self.d.check(seg) or self.d.check(seg.title())) and len(seg) > 3:
                vec[0] = 1.0
            elif len(seg) > 3:
                vec[1] = 1.0
            if count[seg] > threshold:
                vec[index[seg]+2] = 1.0
            seg2vec[seg] = vec

        # segmentation2vec
        self.segmentation2vec = {}
        for segmentation in segmentations:
            f = zeros((self.N))
            for segment in segmentation:
                f += seg2vec[segment]
            self.segmentation2vec[' '.join(segmentation)] = f

    def featurize(self, data, train=False):
        """ featurize """

        featurized_data = []
        for i, (word, ss) in enumerate(data.items()):
            if word in self.data_dev and train:
                continue
            if i == self.train_size and train:
                break
            F = zeros((200, self.N))
            weights = zeros((200))
            forms = [''] * 200
            correct = -1
            for i, (s, w) in enumerate(ss):
                split = [x.strip() for x in s.split(' ')]
                s = []
                for thing in split:
                    if thing != '':
                        s.append(thing)
                s = ' '.join(s)

                if w == CORRECT:
                    continue

                if s == self.word2gold[word]:
                    correct = i
                
                F[i] = self.segmentation2vec[s]
                forms[i] = s
                weights[i] = self.seg2weight[word][s]

            featurized_data.append((word, F, forms, correct, weights/weights.sum()))
        return featurized_data

    def initialize(self):
        """ train the network """

        self.word2segmentations = dd(list)
        self.word2gold = {}
        self.seg2weight = {}
        forms = []
        max_n = 0
        baseline, skyline, total = 0, 0, 0
        for word, ss in self.data.items():
            correct = None
            seg2weight = {}
            for s, w in ss:
                split = [x.strip() for x in s.split(' ')]
                s = []
                for thing in split:
                    if thing != '':
                        s.append(thing)
                s = ' '.join(s)
                if w == CORRECT:
                    correct = s
                    if s not in seg2weight:
                        seg2weight[s] = 0
                    self.word2gold[word] = s
                else:
                    seg2weight[s] = w
            assert correct is not None
            best, best_weight = None, float("-inf")
            
            for seg, weight in seg2weight.items():
                if weight >= best_weight:
                    best_weight = weight
                    best = seg
                tmp = seg.split(" ")
                if '' in tmp:
                    tmp2 = []
                    for x in tmp:
                        if x != '':
                            tmp2.append(x)
                    tmp = tmp2
                forms.append(tmp)
            # 1 best correct
            if correct == best:
                baseline += 1
            # skyline
            if correct in seg2weight:
                skyline += 1
            total += 1
            max_n = max(max_n, len(seg2weight))
            self.seg2weight[word] = seg2weight

        self.forms = forms

def main(lang, logout, pickle_train_in, pickle_dev_in):

    # logger goes to console and to file
    log = logging.getLogger('')
    log.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)s<>%(levelname)s<>%(message)s")
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)
    
    fh = logging.handlers.RotatingFileHandler(logout, maxBytes=(1048576*5), backupCount=7)
    fh.setFormatter(format)
    log.addHandler(fh)
    
    data_train = pickle.load(open(pickle_train_in, 'rb'))
    data_dev = pickle.load(open(pickle_dev_in, 'rb'))
    ll = Model(lang, data_train, data_dev)
    ll.initialize()
    ll.vectorize(ll.forms)
    train_data, test_data = ll.featurize(ll.data_train, True), ll.featurize(ll.data_dev)
    data = train_data + test_data
    ll.model = LogisticRegression(ll.N, 200)

    logging.info("language: {0}".format(lang))
    logging.info("train data size: {0}".format(len(train_data)))
    logging.info("dev data size: {0}".format(len(test_data)))
    
    dev_size = 1000
    train, test = data[:-dev_size], data[-dev_size:]
    eta = 1.0
    gamma = 0.0001
    
    results_file = io.open(logout + '_RESULTS', 'w', encoding='utf-8')
    for iter_num in xrange(20):
        right, possible, total = 0, 0, 0
        edit_distance = 0.0
        f1 = 0.0
        score = 0.0
        sample_counter = 0
        for datum in test:
            sample_counter += 1
            
            word, F, forms, correct, weights = datum
            if correct >= 0:
                score += ll.model.f(F, weights, correct, gamma)
                decoded = forms[ll.model.decode(F, weights)]
                possible += 1
                if iter_num == 19:
                    results_file.write(str(sample_counter) + ':\n' + "*".join(decoded.split(u" ")) + '\t' + "*".join(forms[correct].split(u" ")) + '\n')
                if decoded == forms[correct]:
                    f1 += 1.0
                    right += 1
                else:
                    edit_distance += editdistance.eval("*".join(decoded.split(u" ")), "*".join(forms[correct].split(u" ")))
                    f1 += f1_score(decoded.split(u' '), forms[correct].split(u' '))
                    
            else:
                # gets it wrong
                pass
            total += 1

        # print skyline
        if iter_num == 0:
            logging.info("skyline, dev acc: {0}".format(possible/total))

        logging.info("iteration {0}, dev ll: {1}".format(*(iter_num, score/total)))
        logging.info("iteration {0}, dev acc: {1}".format(*(iter_num, right/total)))
        logging.info("iteration {0}, dev edit distance: {1}".format(*(iter_num, edit_distance/total)))
        logging.info("iteration {0}, dev f1: {1}".format(*(iter_num, f1/total)))

        # learning rate schedule
        if (iter_num+1) % 10 == 0:
            eta *= 0.1
            
        # randomize the training data
        shuffle(train)
        for datum in train:
             word, F, forms, correct, weights = datum
             tau = 1/1.0
             weights = (weights**tau)/((weights**tau).sum())
             if correct >= 0:
                 ll.model.step(F, weights, correct, gamma, eta)

            
if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--lang', type=str)
    p.add_argument('--log', type=str)
    p.add_argument('--pickle_train', type=str)
    p.add_argument('--pickle_dev', type=str)
    args = p.parse_args()

    main(args.lang, args.log, args.pickle_train, args.pickle_dev)
