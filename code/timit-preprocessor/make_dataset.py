import kaldi_io
import sys
import numpy as np
import glob
import os
import pickle as pkl

if __name__ == "__main__":
    file = sys.argv[1]
    out_file = sys.argv[2]

    timit_folder = "/home/antoine/Documents/projects/asr/gm-hmm/data/timit"
    fname_cb = "TIMIT-61.codebook"

    out = kaldi_io.read_mat_scp(file)

    n = 0
    for _, mat in out:
        n += mat.shape[0]

    d = mat.shape[1]

    X = np.zeros((n, d), dtype=np.float32)
    lengths = []
    keys = []
    seq_start = 0
    out = kaldi_io.read_mat_scp(file)
    for key, mat in out:
        l = mat.shape[0]
        lengths.append(l)
        keys.append(key)
        X[seq_start:seq_start+l] = mat
        seq_start += l

    fe = 16e3
    wsize_ms = 10
    nsamples = len(lengths)
    PHN = [0 for _ in range(nsamples)]
    DATA = [0 for _ in range(nsamples)]

    seq_start = 0
    DATA_TYPE = "TEST" if "test" in out_file else "TRAIN"

    for isample in range(nsamples):

        # Get phone labels
        fname = timit_folder + "/"+DATA_TYPE+"/" + keys[isample].replace("-", "/") + ".PHN"
        #print(isample, "/", nsamples, ":", fname)
        with open(fname, "r") as f:
            lines = f.read().split("\n")[:-1]

        PHN_file_data = np.array(list(map(lambda x: x.split(" "), lines)))
        loc = PHN_file_data[:, :2]
        y = PHN_file_data[:, -1]
        del PHN_file_data
        loc = loc.astype(np.int64)
        x = X[seq_start:seq_start + lengths[isample]]

        feats = np.arange(x.shape[0] * wsize_ms).reshape(-1, wsize_ms)


        loc_ms = loc / fe * 1000
        phones = np.copy(feats).reshape(-1)
        # Clean loc_ms
        if loc_ms[-1][1] < phones[-1]:
            loc_ms[-1][1] = phones[-1]

        if loc_ms[0][0] > 0:
            loc_ms[0][0] = 0

        for iy, irange in enumerate(loc_ms.tolist()):
            phones[int(np.floor(irange[0])):int(np.floor(irange[1]))] = iy

        phones = phones.reshape(-1, wsize_ms)

        # Keep longest phoneme on the frame
        phones_index = [np.argmax(np.bincount(x)) for x in phones.tolist()]
        phones = np.array([y[xx] for xx in phones_index])

        PHN[isample] = phones
        DATA[isample] = x
        seq_start = seq_start + lengths[isample]

    # code phonemes
    if os.path.isfile(fname_cb):
        codebook, = pkl.load(open(fname_cb, "rb"))
    else:
        unique_phns = list(set(sum([x.tolist() for x in PHN], [])))
        codebook = dict([(k, v) for k, v in zip(unique_phns, range(len(unique_phns)))])
        pkl.dump([codebook], open(fname_cb, "wb"))


    codedPHN = [list(map(lambda x: codebook[x], phones)) for phones in PHN]
    for i, phones in enumerate(codedPHN):
        DATA[i] = np.concatenate((np.array(phones).reshape(-1, 1), DATA[i]), axis=1)
    #X_ = np.concatenate(tuple(DATA), axis=0)

    pkl.dump([DATA, keys, lengths, PHN], open(out_file, "wb"))
