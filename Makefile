SHELL=/bin/bash

ifndef nclasses
	nclasses=2
endif


ifndef nepochs
	nepochs=10
endif

ifndef nfeats
	nfeats=13
endif


dep=$(shell echo "\$$""\(MODELS\)""\/%"_class{1..$(nclasses)}.mdlc)

all: runfile
	$(MAKE) -f Makefile_run -s prepare_data
	$(MAKE) -f Makefile_run -s train

runfile:
	cat Makefile_cpy | sed -r 's/\(MODELS\)\/\%.mdl:/\(MODELS\)\/\%.mdl: $(dep)/g' |\
	       	sed -r 's/nepochs=/nepochs=$(nepochs)/g'|\
	       	sed -r 's/nclasses=/nclasses=$(nclasses)/g'|\
	      	sed -r 's/nfeats=/nfeats=$(nfeats)/g'	>   Makefile_run

init: runfile
	$(MAKE) -f Makefile_run -s -k $@

watch:
	$(MAKE) -f Makefile_run -s -k $@

clean:
	$(MAKE) -f Makefile_run -s -k $@
