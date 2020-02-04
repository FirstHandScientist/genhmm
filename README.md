# Powering HMM by Generative Models

Code for our work: [Powering Hidden Markov Model by Neural Network based Generative Models](https://arxiv.org/abs/1910.05744)

### Virtual environment
Create a virtual environment with a python3 interpreter, in the newly created `gm_hmm/` directory.
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

## Dataset preparation
See [README.md](src/timit-preprocessor/README.md) in `src/timit-preprocessor`

## Training
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

- Note 1: number of epochs is here number of checkpoints. One checkpoint consist of multiple expectation maximization steps, which you can configure at default.json.


