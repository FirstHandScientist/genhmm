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
$ git clone https://github.com/FirstHandScientist/gm-hmm.git gm_hmm
```

### Virtual environment
Create a virtual environment with a python3 interpreter, inside a folder called `gm_hmm/` folder for instance
```bash
$ cd gm_hmm
$ virtualenv -p python3.6 pyenv
$ cd ..
```

Add the `gm_hmm/` directory to the path:
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

The logs appear in `log/class...`. you can follow the training with:
```bash
$ make watch
```

- Note 1: Modify the `-j` option on the line: `$(MAKE) -f Makefile_run -j 6 -s $$i;` of `Makefile_cpy`. Use `-j $(nclasses)` to create one job per class.
- Note 2: epochs are here Expectation Maximization steps.
- Note 3: make uses the file `Makefile` to modifiy the file 'Makefile_cpy` to create and call `Makefile_run`.




### TIMIT dataset
Read `src/timit-preprocessor/README` for information on how to process the raw timit dataset.
The processed and labeled time series are anyways available in a compressed form `data/test13.gz` and `data/train13.gz`.
You can decompress them using:
```
$ gunzip -c data/test13.gz > test13.pkl
$ gunzip -c data/train13.gz > train13.pkl
```

### Run example
```
python bin/test.py data/ test13.pkl train13.pkl hparams/test.json
```


