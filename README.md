# Powering HMM by Generative Models
---------------------------------------
## draft todo: manuscript
- [x] GenHMM algorithm flow/steps
- [x] LatHMM algorithm flow/steps
- [ ] Generator shared HMM derivatives/algorithm
- [ ] Planning application design of HMM powered by Generative Models.

## Experiments todo
Priority work:
- [x] Utility for trained model saving and loading
- [ ] Improvement for optimization adn training efficiency
    - [x] Adapt Model running on GPU
    - [ ] The mini-batch compositin of PyTorch-Kaldi can be used by us for batch-size stochastica gradient decent. 
- [x] GenHMM sanity check and testing running
- [x] TensorboardX, training process logging and log-likelihood ploting

Import work:
- [ ] Permutation stratergy:
    - [ ] 1. random, 
    - [ ] 2. reverse, 
    - [ ] 3. 1X1 conv
- [ ] 1. Maximum likelihood decesion is not optimal if risk/panelty is not same for all errors. 2. likelihool of each class may differ too much due to random initialization of neural networks: work around: one linear layer after maximum likelihood

- [ ] Comparions experiments design. Please refer to [ranking on TIMIT](https://paperswithcode.com/sota/speech-recognition-on-timit)
    - [ ] Baseline comparion with GMM/HMM
    - [ ] DNN/HMM, LSTM, RNN state art comparion. Please refer project [Pytorch-Kaldi Project](https://github.com/mravanelli/pytorch-kaldi)
 -[ ] Last concern: do we need extra dataset for experiments?
 
 ## Long-term todo
 Potential application to voice translation, song-sing App on movible phones...

## Additional stratergy for improvement on practical experiments
What if GenHMM does not work as good as our expectation?
How about the layer-wise input of mel-spectrograms? Refer to [WaveGlow](https://arxiv.org/abs/1811.00002)

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

Start by creating the necessary folders with:

```bash
$ make init
```

To run the training of genHMM on 2 classes and during 10 epochs, run:
```
$ make nclasses=2 nepochs=10 
```
The training is distributed over two jobs at max.
The parallelization is managed in `Makefile_cpy`.

Modify the `-j` option on the line: `$(MAKE) -f Makefile_run -j 6 -s $$i;` of `Makefile_cpy`. Use `-j $(nclasses)` to create one job per class.

The logs appear in `log/class...`. you can follow the training with:
```bash
$ make watch
```

- Note 1: epochs are here Expectation Maximization steps.
- Note 2: GNU make uses the file `Makefile` to modify the file `Makefile_cpy` and write the modifications to `Makefile_run.


## Dataset preparation
### Dependencies
You must have downloaded the [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) dataset.
You must have a compiled version of [Kaldi](https://github.com/kaldi-asr/kaldi).
This data preparation has been borrowed from [timit-preprocessor](https://github.com/orbxball/timit-preprocessor).
`make_dataset.py` relies on [kaldi-io-for-python](https://github.com/vesis84/kaldi-io-for-python)


### Steps
The steps are as follows:
```bash
$ cd src/timit-preprocessor
$ ./convert_wav.sh
$ python parsing.py  path_to_TIMIT/ test
$ python parsing.py  path_to_TIMIT/ train
$ ./extract_mfcc.sh path_to_KALDI/ path_to_TIMIT/ test.39.scp
$ ./extract_mfcc.sh path_to_KALDI/ path_to_TIMIT/ train.39.scp
$ python make_dataset.py test.39.scp path_to_TIMIT/ test39.pkl
$ python make_dataset.py train.39.scp path_to_TIMIT/ train39.pkl
```

These steps create two files: `test39.pkl` and `train39.pkl`

Note 1: Replace 39 with 13 to have the 13 MFCCs without deltas and delta-deltas.

