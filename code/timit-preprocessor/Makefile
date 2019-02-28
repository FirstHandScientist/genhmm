#/bin/bash
PROJECT_ROOT=/home/antoine/Documents/projects/asr/
KALDI_ROOT=$(PROJECT_ROOT)/kaldi
TIMIT_ROOT=$(PROJECT_ROOT)/gm-hmm/data/TIMIT


all: test.13.pkl train.13.pkl train.39.cmvn.pkl test.39.cmvn.pkl

%.pkl:%.scp
	python make_dataset.py $^

%.scp:
	python parsing.py $(TIMIT_ROOT) train
	python parsing.py $(TIMIT_ROOT) test
	./extract_mfcc.sh $(KALDI_ROOT)

