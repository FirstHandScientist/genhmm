# GenerativeModel-HMM implementation
#
#
SHELL=/bin/bash
PYTHON=python

SRC=src
BIN=bin
DATA_tmp=data
MODELS_tmp=models

ifndef totclasses
	totclasses=61
endif

ifndef nepochs
	nepochs=10
endif
ifndef nclasses
	nclasses=2
endif
ifndef nfeats
	nfeats=39
endif
ifndef j
	j=2
endif

ifndef model
	model=gaus
endif
ifndef exp_name
	exp_name=default
endif
ifndef noise_std
	noise_std=0
endif
ifndef tepoch
	tepoch=1
endif

ROBUST=robust
EXP=exp/$(model)
EXP_DIR=$(EXP)/$(nfeats)feats/$(exp_name)

MODELS=models
DATA=data
LOG=log
init: MODELS=$(EXP_DIR)/models
init: LOG=$(EXP_DIR)/log


MODELS_INTERM=$(shell echo $(MODELS)/epoch{1..$(nepochs)})

training_data=$(DATA)/train.$(nfeats).pkl
testing_data=$(DATA)/test.$(nfeats).pkl

mdl_dep=$(shell echo $(MODELS)/%_class{1..$(nclasses)}.mdlc)
acc_dep=$(shell echo $(MODELS)/%_class{1..$(nclasses)}.accc)
rbst_dep=$(shell echo $(ROBUST)/epoch$(tepoch)_class{1..$(nclasses)}.accc)



all: train


test:
	echo $(mdl_dep)
	echo $(acc_dep)
	echo ${rbst_dep}



init:
	mkdir -p $(MODELS) $(LOG)
	ln -s $(realpath data) $(EXP_DIR)/data
	ln -s $(realpath bin) $(EXP_DIR)/bin
	ln -s $(realpath src) $(EXP_DIR)/src
	cp default.json $(EXP_DIR)
	sed -e 's/model=.*/model=$(model)/' -e 's/nfeats=.*/nfeats=${nfeats}/' -e 's/totclasses=.*/totclasses=$(totclasses)/' Makefile > $(EXP_DIR)/Makefile


prepare_data: $(training_data) $(testing_data)
	$(PYTHON) $(BIN)/prepare_data.py "$(nclasses)/$(totclasses)" $^

train: prepare_data
	echo $(DATA) $(MODELS) $(LOG)
	echo $(MODELS_INTERM)
	for i in $(MODELS_INTERM); do \
		if [[ `echo $${i%.*}_class*.mdlc | wc -w` != $(nclasses) ]]; then rm -f $$i.{mdl,acc}; fi; \
		$(MAKE) -j $(j) -s $$i.mdl; \
	 	$(MAKE) -j $(j) -s $$i.acc; \
	 	sleep 2;\
	done
#	echo "Done" > $^


$(MODELS)/%.mdl: $(mdl_dep)
	$(PYTHON) $(BIN)/aggregate_models.py $@

$(MODELS)/%.acc: $(acc_dep)
	$(PYTHON) $(BIN)/aggregate_accuracy.py $(training_data) $(testing_data) $^ > $@
	cat $@ >> $(LOG)/class_all.log


$(MODELS)/%.mdlc:
	$(eval logfile=$(LOG)/`basename $@ | sed -e 's/^.*\(class\)/\1/g' -e 's/.mdlc/.log'/g`)
	echo `date` ":" $(PYTHON) $(BIN)/train_class_$(model).py $(training_data) $@ >> $(logfile)
	$(PYTHON) $(BIN)/train_class_$(model).py $(training_data) $@ >> $(logfile)

$(MODELS)/%.accc: $(MODELS)/%.mdlc
	$(PYTHON) $(BIN)/compute_accuracy_class.py $^ $(training_data) $(testing_data) >> $@


noise_test: 
	$(MAKE) -j $(j) -s $(ROBUST)/epoch$(tepoch).acc


$(ROBUST)/epoch$(tepoch).acc: $(rbst_dep)
	$(PYTHON) $(BIN)/aggregate_accuracy.py $(training_data) $(testing_data) $^ > $@

$(ROBUST)/%.accc:
	@echo $(subst $(ROBUST),$(MODELS),$@)
	$(PYTHON) $(BIN)/compute_accuracy_class.py $(subst .accc,.mdlc,$(subst $(ROBUST),$(MODELS),$@)) $(training_data) $(testing_data) $(noise_std) >> $@


watch:
	tail -f $(LOG)/class*.log

clean:
#	rm -f $(DATA)/train*_*.pkl
#	rm -f $(DATA)/test*_*.pkl 
#	rm -f $(DATA)/class_map.json
	rm -f $(MODELS)/epoch*.{mdl,acc} 
	rm -f $(MODELS)/epoch*_class*.{mdlc,accc}
	rm -f $(LOG)/class*.log

clean-data:
	rm -f $(DATA)/*_*.pkl $(DATA)/class_map.json


.SECONDARY: 

.PRECIOUS:
