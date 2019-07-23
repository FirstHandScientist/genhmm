#/bin/bash
SHELL=/bin/bash

######


PROJECT_ROOT=/home/antoine/Documents/projects/asr
KALDI_ROOT=$(PROJECT_ROOT)/kaldi
TIMIT_ROOT=$(PROJECT_ROOT)/gm_hmm/data/TIMIT
DATA_OUT=$(PROJECT_ROOT)/gm_hmm/data


#####

targets_= $(shell echo {test,train}.{13,39}.pkl)
targets= $(addprefix $(DATA_OUT)/,$(targets_))

interm=.data
material=$(interm)/material
processed=$(interm)/processed

interm_files= $(shell echo $(material)/{train,test}.lbl)

all: $(targets)
	

%.pkl: %.scp
	python make_dataset.py $^ $(TIMIT_ROOT) $@

%.scp: $(interm_files)
	./extract_mfcc.sh $(KALDI_ROOT) $(TIMIT_ROOT) $(processed) $@

$(material)/%.lbl:
	python parsing.py $(TIMIT_ROOT) $*

convert:
	./convert_wav.sh $(TIMIT_ROOT) $(KALDI_ROOT)

clean:
	rm -rf $(interm)
	rm -f $(DATA_OUT)/*.{scp,pkl}

.SECONDARY:
