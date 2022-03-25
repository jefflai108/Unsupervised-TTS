#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
n_fft=2048
n_shift=300
win_length=1200

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=alex_train-clean-460_nopunc
valid_set=alex_dev-clean_nopunc
test_sets="alex_dev-clean_nopunc alex_test-clean_nopunc"

train_config=conf/tuning/sls_train_gst+xvector_tacotron2.yaml
inference_config=conf/decode.yaml

local_data_opts="--trim_all_silence true" # trim all silence in the audio

./tts-nopunc.sh \
    --ngpu 4 \
    --lang en \
    --feats_type raw \
    --local_data_opts "${local_data_opts}" \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --cleaner none \
    --g2p none \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --use_xvector true \
    --xvector_tool speechbrain \
    --use_sid false \
    --use_lid false \
    --stage 6 --stop-stage 6 \
    --tag sls_train_gst+xvector_tacotron2_raw_phn_none-nopunc \
    --tts_stats_dir exp/tts_stats_raw_phn_none-nopunc \
    ${opts} "$@"
