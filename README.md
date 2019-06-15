# Powering HMM by Generative Models
---------------------------------------
## draft todo: manuscript
- [x] GenHMM algorithm flow/steps
- [x] LatHMM algorithm flow/steps
- [ ] Generator shared HMM derivatives/algorithm
- [ ] Planning application design of HMM powered by Generative Models.

## Experiments todo
Priority work:
- [ ] Utility for trained model saving and loading
- [ ] Improvement for optimization adn training efficiency
    - [ ] Adapt Model running on GPU
    - [ ] The mini-batch compositin of PyTorch-Kaldi can be used by us for batch-size stochastica gradient decent. 
- [ ] GenHMM sanity check and testing running
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
To install hmmlearn library, run:
 
```
$ pip install --upgrade --user hmmlearn
```
or
```
$ git clone  https://github.com/hmmlearn/hmmlearn.git src/
$ cd src/hmmlearn
$ python setup.py install
```
Go to https://github.com/hmmlearn/hmmlearn for details.

Create a conda environment from the `.yml` file:
```
$ conda env create -f environment.yml
$ conda activate pyasr
```

Run the code from this README.md location and start by creating the necessary folders with:
```
$ make init
```

To run the training of genHMM on 2 classes and during 10 epochs, run:
```
$ make nclasses=2 nepochs=10 train
```

You can follow the training in `stdout`.
Note 1: make automatically creates as many jobs as the number of classes.
Note 2: epochs are here Expectation Maximization steps.
Note 3: make uses the file `Makefile` to modifiy the file 'Makefile_cpy` to create and call `Makefile_run`.



## Getting Started
The training is distributed over the number of classes on different processes.
The parallelization is managed via a Makefile.
Data and models are pushed to available devices in the `GenHMM.fit()` method.


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


