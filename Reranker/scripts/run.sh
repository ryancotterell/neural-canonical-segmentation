#!/bin/bash

python ncs/loglin.py --pickle_train data/train_answers_new_english.pkl --pickle_dev data/dev_answers_new_english.pkl --lang en --log results/english &
python ncs/loglin.py --pickle_train data/train_answers_new_german.pkl --pickle_dev data/dev_answers_new_german.pkl --lang de --log results/german &
python ncs/loglin.py --pickle_train data/train_answers_new_indonesian.pkl --pickle_dev data/dev_answers_new_indonesian.pkl --lang id --log results/indonesian &
 
