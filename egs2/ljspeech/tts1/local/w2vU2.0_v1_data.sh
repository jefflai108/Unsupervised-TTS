#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

ALEX_DIR=/data/sls/scratch/alexhliu/w2vu/intspch_exp/lj/decode/best/kaldi/jeff-preprocess

for data_split in test valid train; do 
    target_data_dir=data/w2vU2.0_v1_${data_split}
    mkdir -p $target_data_dir
    cp ${ALEX_DIR}/${data_split}.phn ${target_data_dir}/text
    python src/create_data_from_alex.py \
        --output_data_dir $target_data_dir \
        --alex_data_dir $ALEX_DIR --data_split $data_split

    utt2spk=${target_data_dir}/utt2spk
    spk2utt=${target_data_dir}/spk2utt
    utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}

    utils/fix_data_dir.sh $target_data_dir
    utils/validate_data_dir.sh --no-feats $target_data_dir
done 

log "Successfully finished. [elapsed=${SECONDS}s]"
