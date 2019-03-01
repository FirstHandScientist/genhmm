#!/usr/bin/env python

import unittest
import sys

import kaldi_io

class CnetReadTest(unittest.TestCase):
    def testCnetRead(self):
        """ Test reading a 'confusion network', represented by same Kaldi type as Posterior, """

        cnet_f='tests/data/1.cnet'
        cntime_f='tests/data/1.cntime'

        # just read, no regression test,
        cnet = [ (k,v) for k,v in kaldi_io.read_cnet_ark(cnet_f) ]
        cntime = [ (k,v) for k,v in kaldi_io.read_cntime_ark(cntime_f) ]

