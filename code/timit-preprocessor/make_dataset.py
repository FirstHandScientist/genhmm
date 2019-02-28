import kaldi_io
import sys
import numpy as np
import glob
import os
import pickle as pkl

if __name__ == "__main__":
    file = sys.argv[1]
    out_file = sys.argv[2]

    out = kaldi_io.read_mat_scp(file)

    n = 0
    for _, mat in out:
        n += mat.shape[0]

    d = mat.shape[1]

    X = np.zeros((n, d), dtype=np.float32)
    lengths = []
    keys = []
    seq_start = 0

    for key, mat in out:
        l = mat.shape[0]
        lengths.append(l)
        keys.append(key)
        X[seq_start:seq_start+l] = mat

    pkl.dump([X, lengths, keys], open(out_file, "wb"))
    sys.exit(0)
