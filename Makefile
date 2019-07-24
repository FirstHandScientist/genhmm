# GenerativeModel-HMM implementation
#
#
SHELL=/bin/bash
PYTHON=python

SRC=src
BIN=bin
DATA_tmp=data
MODELS_tmp=models
LOG=log

ifndef nepochs
	nepochs=10
endif
ifndef nclasses
	nclasses=2
endif
ifndef nfeats
	nfeats=13
endif
ifndef j
	j=2
endif


DATA=$(nfeats)feats/data
MODELS=$(nfeats)feats/models
LOG=$(nfeats)feats/log
MODELS_INTERM=$(shell echo $(MODELS)/epoch{1..$(nepochs)})

training_data=$(DATA)/train.$(nfeats).pkl
testing_data=$(DATA)/test.$(nfeats).pkl

mdl_dep=$(shell echo $(MODELS)/%_class{1..$(nclasses)}.mdlc)
acc_dep=$(shell echo $(MODELS)/%_class{1..$(nclasses)}.accc)

test:
	echo $(mdl_dep)
	echo $(acc_dep)


all: train

init:
	mkdir -p $(MODELS) $(LOG) $(DATA)


prepare_data: $(training_data) $(testing_data)
	$(PYTHON) $(BIN)/prepare_data.py $(nclasses) $^

train: prepare_data 
	for i in $(MODELS_INTERM); do \
		if [[ `echo $${i%.*}_class* | wc -w` != $(nclasses) ]]; then rm -f $$i; fi; \
		$(MAKE) -n -j $(j) -s $$i.mdl; \
		$(MAKE) -n -j $(j) -s $$i.acc; \
		sleep 2;\
	done
#	echo "Done" > $^


$(MODELS)/%.mdl: $(mdl_dep)
	$(PYTHON) $(BIN)/aggregate_models.py $@

$(MODELS)/%.acc: $(mdl_dep)
	$(PYTHON) $(BIN)/aggregate_accuracy.py $(training_data) $(testing_data) $^ >> $(LOG)/class_all.log


$(MODELS)/%.mdlc:
	$(eval logfile=$(LOG)/`basename $@ | sed -e 's/^.*\(class\)/\1/g' -e 's/.mdlc/.log'/g`)
	echo `date` ":" $(PYTHON) $(BIN)/train_class_gaus.py $(training_data) $@ >> $(logfile)
	$(PYTHON) $(BIN)/train_class_gaus.py $(training_data) $@ >> $(logfile)

$(MODELS)/%.accc: $(MODELS)/%.mdlc
	$(PYTHON) $(BIN)/compute_accuracy_class.py $^ $(training_data) $(testing_data) >> $@

watch:
	tail -f $(LOG)/class*.log

clean:
	rm -f $(DATA)/train*_*.pkl
	rm -f $(DATA)/test*_*.pkl 
	rm -f $(DATA)/class_map.json
	rm -f $(MODELS)/epoch*.{mdl,acc} 
	rm -f $(MODELS)/epoch*_class*.{mdlc,accc}
	rm -f $(LOG)/class*.log


.SECONDARY: 

.PRECIOUS:
