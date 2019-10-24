# GenerativeModel-HMM implementation
#

SHELL=/bin/bash
PYTHON=OPENBLAS_NUM_THREADS=10 pyenv/bin/python

SRC=src
BIN=$(abspath bin)

ifndef paramfile
	paramfile=default.json
endif

ifndef nepochs
	nepochs=10
endif

ifndef nclasses
	nclasses=2
endif

ifndef j
	j=2
endif

ifndef EXP
	EXP=exp
endif

ifndef model
	model=gaus
endif

ifndef model_dir
	model_dir=$(model).$(shell echo $(paramfile) | sed 's/.json//g')
endif

ifndef DATA
	DATA=$(abspath $(EXP)/data)
endif

ifndef MODELS
	MODELS=$(EXP)/models/$(model_dir).$(dataname_trunk)
endif

ifndef LOG
	LOG=$(EXP)/log/$(model_dir).$(dataname_trunk)
endif
ifndef dataname_trunk
	dataname_trunk=39
endif
ifndef training_data
	training_data=$(DATA)/train.$(dataname_trunk).pkl
endif

ifndef testing_data
	testing_data=$(DATA)/test.$(dataname_trunk).pkl
endif

MODELS_INTERM=$(shell echo $(MODELS)/epoch{1..$(nepochs)})


mdl_dep=$(shell echo $(MODELS)/%_class{1..$(nclasses)}.mdlc)
acc_dep=$(shell echo $(MODELS)/%_class{1..$(nclasses)}.accc)

all: train

init:
	mkdir -p $(EXP) $(MODELS) $(LOG)

test:
	echo $(mdl_dep)
	echo $(acc_dep)

train: init
	@for i in $(MODELS_INTERM); do \
		if [[ `echo $${i%.*}_class*.mdlc | wc -w` != $(nclasses) ]]; then rm -f $$i.{mdl,acc}; fi; \
		$(MAKE) -j $(j) $$i.mdl; \
	 	$(MAKE) -j $(j) $$i.acc; \
	 	sleep 2;\
	done

$(MODELS)/%.mdl: $(mdl_dep)
	$(PYTHON) $(BIN)/aggregate_models.py $@ $(paramfile)

$(MODELS)/%.acc: $(acc_dep)
	$(PYTHON) $(BIN)/aggregate_accuracy.py $(training_data) $(testing_data) $^ > $@
	cat $@ >> $(LOG)/class_all.log


$(MODELS)/%.mdlc:
	$(eval logfile=$(LOG)/`basename $@ | sed -e 's/^.*\(class\)/\1/g' -e 's/.mdlc/.log'/g`)
	@echo `date` ":" $(PYTHON) $(BIN)/train_class_$(model).py $(training_data) $@ $(paramfile)>> $(logfile)
	$(PYTHON) $(BIN)/train_class_$(model).py $(training_data) $@ $(paramfile) >> $(logfile)

$(MODELS)/%.accc: $(MODELS)/%.mdlc
	$(PYTHON) $(BIN)/compute_accuracy_class.py $^ $(training_data) $(testing_data) >> $@

# testing part only
# test one checkpoint

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
