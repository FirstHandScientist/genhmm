#!/usr/bin/env bash

# Only need to convert once in the beginning
# Because this program will replace the original wav file (NIST format) into RIFF format in-place

if [ ${#} != 2 ];then
    echo "Usage: ./convert_wav.sh [TIMIT folder's relative path] [Kaldi folder's relative path]"
    exit -1
fi

#data=~/Workspace/data/timit
#kaldi=~/Workspace/kaldi/
data=${1}
kaldi=${2}

find ${data} -name "*.wav" | xargs -n 1 -I '{}' ${kaldi}/tools/sph2pipe_v2.5/sph2pipe -f wav '{}' '{}'.tmp
find ${data} -name "*.wav" | xargs -n 1 -I '{}' mv '{}'.tmp '{}'
