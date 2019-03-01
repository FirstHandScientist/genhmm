#!/usr/bin/env bash

if [ ${#} != 1 ];then
    echo "Usage: ./mfcc_extract.sh [Kaldi's relative path]"
    exit -1
fi

kaldi=${1}
path=data/processed
options="--use-energy=false"

mkdir -p $path


echo "Extracting train set"
target=train
log=$path/$target.extract.log

${kaldi}/src/featbin/compute-mfcc-feats --verbose=2 $options scp:data/material/$target.wav.scp ark,t,scp:$path/$target.13.ark,$path/$target.13.scp 2> $log
${kaldi}/src/featbin/add-deltas ark:$path/$target.13.ark ark,t:$path/$target.13_deltas.ark
${kaldi}/src/featbin/compute-cmvn-stats ark:$path/$target.13_deltas.ark ark,t:$path/$target.cmvn_results.ark
${kaldi}/src/featbin/apply-cmvn ark:$path/$target.cmvn_results.ark ark:$path/$target.13_deltas.ark ark,t,scp:$path/$target.39.cmvn.ark,$path/$target.39.cmvn.scp


echo "Extracting test set"
target=test
log=$path/$target.extract.log

${kaldi}/src/featbin/compute-mfcc-feats --verbose=2 $options scp:data/material/$target.wav.scp ark,t,scp:$path/$target.13.ark,$path/$target.13.scp 2> $log
${kaldi}/src/featbin/add-deltas ark:$path/$target.13.ark ark,t:$path/$target.13_deltas.ark
${kaldi}/src/featbin/compute-cmvn-stats ark:$path/$target.13_deltas.ark ark,t:$path/$target.cmvn_results.ark
${kaldi}/src/featbin/apply-cmvn ark:$path/$target.cmvn_results.ark ark:$path/$target.13_deltas.ark ark,t,scp:$path/$target.39.cmvn.ark,$path/$target.39.cmvn.scp
