#/bin/bash
SHELL=/bin/bash
PROJECT_ROOT=/home/antoine/Documents/projects/asr
KALDI_ROOT=$(PROJECT_ROOT)/kaldi
TIMIT_ROOT=$(PROJECT_ROOT)/gm-hmm/data/TIMIT
DATA_OUT=$(PROJECT_ROOT)/gm-hmm/data
targets_= test.13.pkl train.13.pkl train.39.cmvn.pkl test.39.cmvn.pkl

targets= $(addprefix $(DATA_OUT)/,$(targets_))
interm=.data/material

interm_files=$(interm)/test.lbl $(interm)/train.lbl

	
all: $(targets)
	

%.pkl: %.scp
	python make_dataset.py $^ $(interm) $@

%.scp: $(interm_files)
	./extract_mfcc.sh $(KALDI_ROOT) $(TIMIT_ROOT) $@

$(interm)/%.lbl:
	python parsing.py $(TIMIT_ROOT) $*

.SECONDARY:
