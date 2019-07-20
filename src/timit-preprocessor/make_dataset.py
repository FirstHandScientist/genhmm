import kaldi_io
import sys
import numpy as np
import glob
import os
import pickle as pkl
import argparse


def test_make_phone_index_sequence():
    a = np.array([[0, 20], [20, 36], [36, 45]])
    expected = np.arange(a.shape[0])
    expected[0:20] = 0
    expected[20:36] = 1
    expected[36:45] = 1
    assert((make_phone_index_sequence(a) == expected).all())


def make_phone_index_sequence(loc_ms, utt_end):
    out = np.arange(utt_end)
    for iy, irange in enumerate(loc_ms.tolist()):
        out[int(np.floor(irange[0])):int(np.floor(irange[1]))] = iy
    return out


def test_make_range_match():
    a = np.array([[10, 20], [20, 36], [36, 43]])
    assert(make_range_match(a, 45) == np.array([[0, 20], [20, 36], [36, 45]]))


def make_range_match(loc_ms, utt_ending):
    # In case the ending of the last phoneme is before the ending of our feature sequence:
    # push the end of the last phoneme.
    if loc_ms[-1][1] < utt_ending:
        loc_ms[-1][1] = utt_ending

    # If the first phoneme does not exactly start at 0,
    # Pull the begining of the first phoneme
    if loc_ms[0][0] > 0:
        loc_ms[0][0] = 0
    return loc_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform time labels into MFCC labels.\n"\
                                                  "Example: python make_dataset.py test.39.scp ../../TIMIT test39.pkl")

    parser.add_argument('infile', metavar="<Input scp file>", type=str)
    parser.add_argument('timit_folder', metavar="<path to timit>", type=str)
    parser.add_argument('outfile', metavar="<Output .pkl file>", type=str)
    args = parser.parse_args()


    infile = args.infile
    timit_folder = args.timit_folder
    outfile = args.outfile


    timit_folder = "/home/antoine/Documents/projects/asr/gm-hmm/data/timit"
    fname_cb = "TIMIT-61.codebook"

    seq_start = 0

    # Now we must have the phonemes corresponding the the MFCC for each utterance
    fe = 16e3

    # That's the step size used to compute the MFCC
    wsize_ms = 10

    lengths = [] # Size of all sequnces
    keys = [] # Name of the utterance
    PHN = [] # List of label sequences
    DATA = [] # List of mfcc sequnces

    DATA_TYPE = "TEST" if "test" in outfile else "TRAIN"

    # Read mfccs
    out = kaldi_io.read_mat_scp(infile)

    # Each utterance
    for key, mat in out:
        # get the length and store key
        l = mat.shape[0]
        lengths.append(l)
        keys.append(key)

        # Get phone labels for the utterance
        fname = os.path.join(timit_folder, DATA_TYPE, keys[-1].upper().replace("-", "/") + ".PHN")

        # lines is a list of strings, each string contains the temporal range of a phoneme, with unit in samples.
        with open(fname, "r") as f:
            lines = f.read().split("\n")[:-1]

        # Here we simply reformat the lines to have the location information in an array
        PHN_file_data = np.array(list(map(lambda x: x.split(" "), lines)))

        # Range of the phoneme
        label_location_sample = PHN_file_data[:, :2]

        # Actual Phoneme name
        label_name = PHN_file_data[:, -1]

        label_location_sample = label_location_sample.astype(np.int64)
        utt_end = l * wsize_ms
        label_location_ms = label_location_sample / fe * 1000
        label_location_ms = make_range_match(label_location_ms, utt_end)

        # label_seq_ms [ 0 0 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3]
        label_seq_ms = make_phone_index_sequence(label_location_ms,utt_end)

        # label_seq_ms=[[ 0 0 1 1 1 1 1 1 1 2 2 2],
        #  [ 2 2 2 2 2 2 2 2 2 2 2 2],
        #  [ 2 3 3 3 3 3 3 3 3 3 3 3 3]
        label_seq_ms = label_seq_ms.reshape(-1, wsize_ms)

        # label_seq_mfcc = [1,2,3]
        label_seq_mfcc = [np.argmax(np.bincount(x)) for x in label_seq_ms.tolist()]
        label_seq_mfcc_names = np.array([label_name[xx] for xx in label_seq_mfcc])

        assert(len(label_seq_mfcc_names) == l)

        PHN.append(label_seq_mfcc_names)
        DATA.append(mat)





    # Now create a reference codebook or load the existing one
    if os.path.isfile(fname_cb):
        codebook, = pkl.load(open(fname_cb, "rb"))

    else:
        # Find the unique list of phonemes
        unique_phns = list(set(sum([x.tolist() for x in PHN], [])))

        # Create a codebook {phoneme1: 0, phoneme2: 1, ... }
        codebook = dict([(k, v) for k, v in zip(unique_phns, range(len(unique_phns)))])

        # Save to file
        pkl.dump([codebook], open(fname_cb, "wb"))

    # Code the phones with the codebook
    codedPHN = [list(map(lambda x: codebook[x], phones)) for phones in PHN]

    for i, phones in enumerate(codedPHN):
        # Concatenate the sequence of phoneme to the data
        DATA[i] = np.concatenate((np.array(phones).reshape(-1, 1), DATA[i]), axis=1)

    # Store the obtained data, where the test data contain the codebook.
    if "test" in os.path.basename(outfile):
        pkl.dump([DATA, keys, lengths, codebook, PHN], open(outfile, "wb"))
    else:
        pkl.dump([DATA, keys, lengths, PHN], open(outfile, "wb"))

    sys.exit(0)