# TIMIT Preprocessor

**timit-preprocessor** extract mfcc vectors and phones from TIMIT dataset for advanced use on speech recognition.

## Overview
The TIMIT corpus of read speech is designed to provide speech data for acoustic-phonetic studies and for the development and evaluation of automatic speech recognition systems. More information on [website](https://catalog.ldc.upenn.edu/ldc93s1) or [Wiki](https://en.wikipedia.org/wiki/TIMIT)

## Installation

Note that to install [Kaldi](https://github.com/kaldi-asr/kaldi) first by following the instructions in [`INSTALL`](https://github.com/kaldi-asr/kaldi/blob/master/INSTALL).

> (1)  
> go to tools/ and follow INSTALL instructions there.  
>
> (2) 
> go to src/ and follow INSTALL instructions there.  

After running the scripts instructed by `INSTALL` in `tools/`, there will be reminder as followed. Go and run it.

> Kaldi Warning: IRSTLM is not installed by default anymore. If you need IRSTLM, use the script `extras/install_irstlm.sh`

After ensuring kaldi installation, we can start by running

```
git clone https://github.com/Jy-Liu/timit-preprocessor.git
```

## Preprocessing

### Steps

1. Run `./convert_wav.sh` only in the **first time** after cloning this repo.

2. `python3 parsing.py -h` to see instructions parsing timit dataset for phone labels and raw intermediate files in folder `data/material/`.

3. `./extract_mfcc.sh` to extract mfcc vectors into .scp and .ark files.

Finally, there's a folder called `data/` which contains all the outcomes in the belowing directory structure:

```
data/
|-- material
|   |-- test.lbl
|   `-- train.lbl
`-- processed
    |-- test.39.cmvn.ark
    |-- test.39.cmvn.scp
    |-- test.extract.log
    |-- train.39.cmvn.ark
    |-- train.39.cmvn.scp
    `-- train.extract.log
```

If you want to do further operations, there's a good repo called [kaldi-io-for-python](https://github.com/vesis84/kaldi-io-for-python).

## Contact
Feel free to [contact me](mailto:junyouliu9@gmail.com) if there's any problems.

### License

BSD 3-Clause License (2017), Jun-You Liu
