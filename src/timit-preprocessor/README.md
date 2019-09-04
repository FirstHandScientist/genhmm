# TIMIT Preprocessor

**timit-preprocessor** extract mfcc vectors and phones from TIMIT dataset for advanced use on speech recognition.

## Overview
The TIMIT corpus of read speech is designed to provide speech data for acoustic-phonetic studies and for the development and evaluation of automatic speech recognition systems.
More information on [website](https://catalog.ldc.upenn.edu/ldc93s1) or [Wiki](https://en.wikipedia.org/wiki/TIMIT)
The instructions and scripts used here are built upon [timit-preprocessor](https://github.com/orbxball/timit-preprocessor).
`make_dataset.py` relies on [kaldi-io-for-python](https://github.com/vesis84/kaldi-io-for-python)


## Dependencies
You must have downloaded the [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) dataset.
You must have a compiled version of [Kaldi](https://github.com/kaldi-asr/kaldi).

Note that to install [Kaldi](https://github.com/kaldi-asr/kaldi) first by following the instructions in [`INSTALL`](https://github.com/kaldi-asr/kaldi/blob/master/INSTALL).

> (1)  
> go to tools/ and follow INSTALL instructions there.  
>
> (2) 
> go to src/ and follow INSTALL instructions there.  

After running the scripts instructed by `INSTALL` in `tools/`, there will be reminder as followed. Go and run it.

> Kaldi Warning: IRSTLM is not installed by default anymore. If you need IRSTLM, use the script `extras/install_irstlm.sh`

## Preprocessing
### Steps
1. source the python interpreter matching the [requirement.txt](../../requirements.txt) file.
```bash
$ source ../../pyenv/bin/activate
```

2. Edit the default of variables `KALDI_ROOT`, `TIMIT_ROOT`, `DATA_OUT` in the [Makefile](Makefile) to match your installation.
You can also leave the default as is and use `make` with *location arguments*.
```bash
$ make KALDI_ROOT=abc/kaldi  TIMIT_ROOT=abc/timit DATA_OUT=abc/out ...
```

3. Run the following commands (here without *location arguments*):
```bash
$ make convert
$ make -j 4
```

Note 1: noisy `.wav` files will be created alongside timit clean ones.

Note 2: In case of errors, display the remaining steps:
```bash
$ make -n
```
and try to debug them one by one.

Note 3: For serious problems you can always contact us in the [issues] section.

### Acknowledgment
Some codes of the TIMIT Preprocessor are from the following repo: [TIMIT Preprocessor](https://github.com/orbxball/timit-preprocessor)
