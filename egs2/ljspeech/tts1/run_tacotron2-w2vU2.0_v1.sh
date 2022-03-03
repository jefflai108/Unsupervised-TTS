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
test_sets="alex_valid_nopunc alex_test_nopunc" # use ground-truth text for decoding 

train_config=conf/tuning/sls_train_tacotron2.yaml
inference_config=conf/tuning/decode_tacotron2.yaml

# modification based on: https://github.com/espnet/espnet/issues/4109
for inference_model in valid.loss.best valid.loss.ave; do 
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
    --ngpu 2 --stage 7 --stop-stage 7 \
    --tag sls_train_tacotron2_raw_phn_none-w2vU2.0_v1 \
    --tts_stats_dir exp/tts_stats_raw_phn_none-w2vU2.0_v1 \
    --inference_model ${inference_model}.pth \
    --inference_tag decode_tacotron2_${inference_model}_parallel_wavegan.v3 \
    --vocoder_file /data/sls/temp/clai24/pretrained-models/vocoders/train_nodev_ljspeech_parallel_wavegan.v3/checkpoint-3000000steps.pkl \
    ${opts} "$@"
done 

    #--inference_model valid.loss.best.pth \
    #--inference_tag decode_tacotron2_valid.loss.best_parallel_wavegan.v3 \
    #--vocoder_file /data/sls/temp/clai24/pretrained-models/vocoders/train_nodev_ljspeech_parallel_wavegan.v3/checkpoint-3000000steps.pkl \

    #--inference_model valid.loss.ave.pth \
    #--inference_tag decode_tacotron2_valid.loss.ave_parallel_wavegan.v3 \
    #--vocoder_file /data/sls/temp/clai24/pretrained-models/vocoders/train_nodev_ljspeech_parallel_wavegan.v3/checkpoint-3000000steps.pkl \
