#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

speaker=1841_159771
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

train_set=alex_train-clean-460_w2vU2.0_v1-${speaker}-train
valid_set=alex_train-clean-460_w2vU2.0_v1-${speaker}-valid
test_sets="alex_test-clean_nopunc"

train_config=conf/tuning/sls_train_gst+xvector_transformer.yaml
train_config=conf/tuning/sls_8gpu_train_gst+xvector_transformer.yaml # 8 GPU, half the accum_grad
train_config=conf/tuning/sls_8gpu_train_gst+xvector_transformer_spkfinetune.yaml
train_config=conf/tuning/sls_4gpu_train_gst+xvector_transformer_spkfinetune.yaml
inference_config=conf/decode.yaml

local_data_opts="--trim_all_silence true" # trim all silence in the audio
train_args="--init_param exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1/valid.loss.best.pth"

parallel_wavegan_v1=/data/sls/temp/clai24/pretrained-models/vocoders/train_nodev_clean_libritts_parallel_wavegan.v1/checkpoint-1000000steps.pkl
hifigan=/data/sls/temp/clai24/pretrained-models/vocoders/train_nodev_clean_libritts_hifigan.v1/checkpoint-2500000steps.pkl
melgan=/data/sls/temp/clai24/pretrained-models/vocoders/train_nodev_clean_libritts_multi_band_melgan.v2/checkpoint-1000000steps.pkl
vctk_hifigan=/data/sls/temp/clai24/pretrained-models/vocoders/train_nodev_all_vctk_hifigan.v1/checkpoint-2500000steps.pkl

for inference_model in valid.loss.ave; do
./tts-w2vU2.0_v1.sh \
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
    --use_sid true \
    --use_lid false \
    --stage 6 --stop-stage 7 \
    --tag sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1-spkfinetune-${speaker} \
    --tts_stats_dir exp/tts_stats_raw_phn_none-w2vU2.0_v1-spkfinetune-${speaker} \
    --inference_model ${inference_model}.pth \
    --inference_tag decode_transformer_${inference_model}_hifigan \
    --vocoder_file $hifigan \
    --train_args "${train_args}" \
    ${opts} "$@"
done
