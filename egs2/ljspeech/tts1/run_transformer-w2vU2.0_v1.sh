#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=22050
n_fft=1024
n_shift=256

opts=
if [ "${fs}" -eq 22050 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=w2vU2.0_v1_train
valid_set=w2vU2.0_v1_valid
test_sets="w2vU2.0_v1_valid w2vU2.0_v1_test"

train_config=conf/tuning/sls_train_transformer.yaml
inference_config=conf/tuning/decode_transformer.yaml

./tts-w2vU2.0_v1.sh \
    --lang en \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --token_type phn \
    --cleaner none \
    --g2p none \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --ngpu 2 --stage 6 --stop-stage 6 \
    --tag sls_train_transformer_raw_phn_none-w2vU2.0_v1 \
    --tts_stats_dir exp/tts_stats_raw_phn_none-w2vU2.0_v1 \
    ${opts} "$@"
