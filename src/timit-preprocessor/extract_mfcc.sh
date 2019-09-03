#!/usr/bin/env bash

if [ ${#} != 4  ];then
    echo "Usage: ./mfcc_extract.sh [Kaldi's relative path] [TIMIT relative path] [Intermediate folder] [output .scp filename]"
    exit -1
fi

kaldi=${1}
timit=${2}
path=${3}
target_file=${4}

options="--use-energy=false"

mkdir -p $path


fname=$(basename "$target_file")
fname_trunk="${fname%.*}"

log=$path/$target.extract.log
if [[ $fname == *".13."* ]]; then
	target=(${fname_trunk//.13})
	${kaldi}/src/featbin/compute-mfcc-feats --verbose=2 $options scp:.data/material/$target.wav.scp ark,t,scp:$path/$target.13.ark,$target_file 2> $log
fi

if [[ $fname == *".39."* ]]; then
	target=(${fname_trunk//.39})
	${kaldi}/src/featbin/compute-mfcc-feats --verbose=2 $options scp:.data/material/$target.wav.scp ark,t,scp:$path/$target.13.ark,$target_file 2> $log
	${kaldi}/src/featbin/add-deltas ark:$path/$target.13.ark ark,t:$path/$target.13_deltas.ark
	${kaldi}/src/featbin/compute-cmvn-stats ark:$path/$target.13_deltas.ark ark,t:$path/$target.cmvn_results.ark
	${kaldi}/src/featbin/apply-cmvn ark:$path/$target.cmvn_results.ark ark:$path/$target.13_deltas.ark ark,t,scp:$path/$target.39.cmvn.ark,$target_file
fi

exit 0
