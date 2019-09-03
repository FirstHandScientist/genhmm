#/bin/bash
SHELL=/bin/bash

ifndef PROJECT_ROOT
	PROJECT_ROOT=.
endif
ifndef KALDI_ROOT
	KALDI_ROOT=$(PROJECT_ROOT)/kaldi
endif
ifndef DATA_OUT
	DATA_OUT=$(PROJECT_ROOT)/gm_hmm/data
endif
ifndef
	TIMIT_ROOT=$(DATA_OUT)/TIMIT
endif

ifndef ntype
	ntype=clean
endif

ifndef snr
	snr=5
endif


targets_= $(shell echo {test,train}.{13,39}.pkl)
noise_targets_ = $(shell echo test.{13,39}.white.{5,10,15}dB.pkl)

targets= $(addprefix $(DATA_OUT)/,$(targets_)) $(addprefix $(DATA_OUT)/,$(noise_targets_))

interm=.data
material=$(interm)/material
processed=$(interm)/processed

interm_files= $(shell echo $(material)/{train,test}.lbl) $(shell echo $(material)/test.white.{5,10,15}dB.lbl)

all: noise
	
noise: feats
		

feats: $(targets)

%.pkl: %.scp
	python make_dataset.py $^ $(TIMIT_ROOT) $@

%.scp: $(interm_files)
	./extract_mfcc.sh $(KALDI_ROOT) $(TIMIT_ROOT) $(processed) $@

$(material)/%.lbl: %.noise
	python parsing.py $(TIMIT_ROOT) $*

%.noise:
	python add_noise.py -timit $(TIMIT_ROOT) -opt $*
convert:
	./convert_wav.sh $(TIMIT_ROOT) $(KALDI_ROOT)

clean:
	rm -rf $(interm)
	rm -f $(DATA_OUT)/*.{scp,pkl}

.SECONDARY:
