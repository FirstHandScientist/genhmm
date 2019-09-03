from functools import partial
from multiprocessing.dummy import Pool
from scipy.io import wavfile
import argparse
from glob import glob
from parse import parse

import os, sys
import numpy as np


def gen_noise(n, type, sigma, noise_folder="../../data/NoiseDB/NoiseX_16kHz"):
    """Generate noise of a certain type and std."""

    if type == "white":
        noise_filename = os.path.join(noise_folder, "{}_16kHz.wav".format(type))
        _, loaded_noise = wavfile.read(noise_filename)

        try:
            assert(n < loaded_noise.shape[0])
        except AssertionError as e:
            print("Noise file: {} is too short.".format(noise_filename), file=sys.stderr)

        # Find a random section in file.
        istart = np.random.randint(loaded_noise.shape[0] - n)
        raw_noise = loaded_noise[istart:istart+n]

    else:
        print("Unknown {} noise".format(type), file=sys.stderr)
        raw_noise = 0

    return raw_noise / raw_noise.std() * sigma


def new_filename(file, ntype, snr):
    """Append noise tyep and power at the end of wav filename."""
    return file.replace(".WAV", ".WAV.{}.{}dB".format(ntype, snr))


def corrupt_data(s, ntype, snr):
    """Corrupt a signal with a particular noise."""
    s_std = np.std(s)
    n_std = 10 ** (- snr / 20) * s_std
    n = gen_noise(s.shape[0], ntype, n_std)
    sn = (s + n).astype(s.dtype)
    return sn


def corrupt_wav(file, ntype=None, snr=None):
    """Corrupt a wav file with noise and write to a new file."""
    rate, s = wavfile.read(file)
    sn = corrupt_data(s, ntype, snr)
    wavfile.write(new_filename(file, ntype, snr), rate, sn)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a particular noise type to WAV files.")

    parser.add_argument('-timit', metavar="<Timit location>", type=str)
    parser.add_argument('-opt', metavar="<Signal to Noise Ratio (dB)>", type=str)
    parser.add_argument('-j', metavar="<Number of jobs (default: numcpu)>",
                        type=int, default=os.cpu_count())

    args = parser.parse_args()
    try:
        dset, ntype, snr = parse("{}.{}.{:d}dB", args.opt)
    except TypeError as e:
        print("No noise to be added with option: {}.\nExit.".format(args.opt),file=sys.stderr)
        sys.exit(0)

    if dset == "test":
        wavs = glob(os.path.join(args.timit, "TEST", "**" , "*.WAV"), recursive=True)
        f = partial(corrupt_wav, ntype=ntype, snr=snr)
        with Pool(args.j) as pool:
            pool.map(f, wavs)

    sys.exit(0)