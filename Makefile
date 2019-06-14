SHELL=/bin/bash

ifndef nclasses
	nclasses=2
endif


ifndef nepochs
	nepochs=10
endif


dep=$(shell echo "\$$""\(MODELS\)""\/%"_class{1..$(nclasses)}.mdlc)

all:
	echo $(dep)
	cat Makefile_cpy | sed -r 's/\(MODELS\)\/\%.mdl:/\(MODELS\)\/\%.mdl: $(dep)/g' |\
	       	sed -r 's/nepochs=/nepochs=$(nepochs)/g'|\
	       	sed -r 's/nclasses=/nclasses=$(nclasses)/g'  >   Makefile_run

	$(MAKE) -f Makefile_run -s prepare_data
	$(MAKE) -f Makefile_run -s train

clean:
	$(MAKE) -f Makefile_run -s $@

