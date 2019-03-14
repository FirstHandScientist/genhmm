# Powering HMM by Generative Models
---------------------------------------
## draft in dir: manuscript
- [x] GenHMM algorithm flow/steps
- [x] LatHMM algorithm flow/steps
- [ ] Generator shared HMM derivatives/algorithm
- [ ] Planning application design of HMM powered by Generative Models.

## implementation in dir: code
Flow model takes the batch size as a parameter for constructing model structure, we need take this into consideration. 

### Install
To install hmmlearn library (https://github.com/hmmlearn/hmmlearn), run:
```
$ cd code/hmmlearn
$ python setup.py install
```
or 
```
$ pip install --upgrade --user hmmlearn
```
See the repository for details.
### TIMIT dataset
See `src/timit-preprocessor` for information on how to process the raw timit dataset.
The processed and labeled time series are anyways available in `data/test13.pkl` and `data/train13.pkl`.

### Run example
```
python bin/test.py data/ test13.pkl train13.pkl hparams/test.json
```
