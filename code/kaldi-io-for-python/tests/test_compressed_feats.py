#!/usr/bin/env python

import unittest
import sys

import numpy as np
import kaldi_io
import timeit

class CompressedFeatsTest(unittest.TestCase):
    def testReading(self):
        t_beg = timeit.default_timer()
        orig = {k:m for k,m in kaldi_io.read_mat_ark('tests/data/feats.ark')}
        t_read_not_compressed = timeit.default_timer() - t_beg

        t_beg = timeit.default_timer()
        comp = {k:m for k,m in kaldi_io.read_mat_ark('tests/data/feats_compressed.ark')}
        t_read_compressed = timeit.default_timer() - t_beg

        # reading the compressed data should be <5x slower,
        self.assertLess(t_read_compressed, 5.*t_read_not_compressed)

        # check that the values are similar
        # (these are not identical due to discretization in compression),
        for key in orig.keys():
            self.assertGreater(1e-4, np.max(np.abs(comp[key]-orig[key])))

