# Powering HMM by Generative Models

## Installation

First clone the repository:
```bash
$ git clone https://github.com/FirstHandScientist/gm_hmm.git
```

### Virtual environment
Create a virtual environment with a python3 interpreter, inside a folder called `gm_hmm/` folder for instance
```bash
$ cd gm_hmm
$ virtualenv -p python3.6 pyenv
$ cd ..
```

Add the parent directory of `gm_hmm/` to the path:
```bash
$ echo $PWD > gm_hmm/pyenv/lib/python3.6/site-packages/gm_hmm.pth
```

Install the dependencies:
```bash
$ cd gm_hmm
$ source pyenv/bin/activate
$ pip install -r requirements.txt
```
### Additional tools
You must install `GNU make`, on Ubuntu:
```bash
$ sudo apt install build-essential
$ make -v
GNU Make 4.1
Built for x86_64-pc-linux-gnu
...
```

## Getting Started

Start by creating the necessary experimental folders for using model "GenHMM" and data feature length of 39,  with:

```bash
$ make init model=gen nfeats=39 exp_name=genHMM
```
Change directory to the created experiment directory:
```bash
$ cd exp/gen/39feats/genHMM
```
To run the training of genHMM on 2 classes and during 10 epochs, with two distributed jobs, run:
```
$ make j=2 nclasses=2 nepochs=10 
```
Modify the `j` option to change the number of jobs for this experiment.


The logs appear in `log/class...`. you can follow the training with:
```bash
$ make watch
```

- Note 1: epochs are here Expectation Maximization steps.

## Dataset preparation
### Dependencies
You must have downloaded the [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) dataset.
You must have a compiled version of [Kaldi](https://github.com/kaldi-asr/kaldi).
This data preparation has been borrowed from [timit-preprocessor](https://github.com/orbxball/timit-preprocessor).
`make_dataset.py` relies on [kaldi-io-for-python](https://github.com/vesis84/kaldi-io-for-python)

### Steps
First edit the variables `KALDI_ROOT`, `TIMIT_ROOT`, `DATA_OUT` in `src/timit-preprecessor/Makefile`.
Replace the values with the location of kaldi, timit and the place you wish to have the datasets created.

```bash
$ cd src/timit-preprocessor
$ make convert
$ make -j 4
```
