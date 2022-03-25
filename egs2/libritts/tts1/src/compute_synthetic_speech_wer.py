import speech_recognition as sr
import os, sys
import os.path as osp
import jiwer
from jiwer import wer
import string 
import text_cleaners 

def Merge(dict1, dict2):
    return(dict2.update(dict1))

def process_ljspeech():
    total_lj = {}
    #for split in ['test-clean']:
    for split in ['train-clean-460']:
        text = osp.join('/data/sls/scratch/clai24/Unsupervised-TTS/egs2/libritts/tts1/data/', split, 'text')
        with open(text, 'r') as f: 
            df = f.readlines() 
        split_lj = {x.strip('\n').split(' ')[0]:' '.join(x.strip('\n').split(' ')[1:]) for x in df}
        if total_lj is None: 
            total_lj = split_lj 
        else: Merge(split_lj, total_lj)

    # remove final punctuation 
    total_lj = {k:v[:-1] if v[-1] in string.punctuation else v for k,v in total_lj.items()}

    return total_lj

def collect_test_ids(): 
    with open('data/alex_test-clean_nopunc/utt2spk', 'r') as f: 
        target_test_utt2spk = f.readlines() 
    target_test_utt2spk = [x.strip('\n').split()[0] for x in target_test_utt2spk]
    
    return target_test_utt2spk

def recognize_an_audio(audio_path, r, utt2ground_truth, ground_truths, hypotheses):

    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)  # read the entire audio file

    # get the ground truth 
    if '_gen' in audio_path:
        audio_path = audio_path.replace('_gen', '')
    if audio_path.split('/')[-1].split('.wav')[0] not in utt2ground_truth.keys():
        return ground_truths, hypotheses
    ground_truth = utt2ground_truth[audio_path.split('/')[-1].split('.wav')[0]]


    try:
        hypothesis = r.recognize_google(audio, language='en-US')
    except Exception as e:
        ground_truths.append(text_cleaners.english_cleaners(ground_truth))
        hypotheses.append("")
        return ground_truths, hypotheses
    
    ground_truths.append(text_cleaners.english_cleaners(ground_truth))
    hypotheses.append(text_cleaners.english_cleaners(hypothesis))
    #print('ground_truths is', ground_truths)
    #print('hypotheses is', hypotheses)
    return ground_truths, hypotheses

def run(synthesis_directory):

    utt2ground_truth = process_ljspeech()
    #target_uttids = collect_test_ids()
    ground_truths, hypotheses = [], []
    r = sr.Recognizer()
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.Strip(),
    ]) 

    cnt = 0
    for files in os.walk(synthesis_directory):
        roodir = files[0]
        for filename in files[2]:
            file_id = filename.split('.')[0]
            #if filename.endswith(".wav") and file_id in target_uttids:
            if filename.endswith(".wav"):
                audio_file = os.path.join(roodir, filename)
                print('processing %s' % audio_file)
                ground_truths, hypotheses = recognize_an_audio(audio_file, r, utt2ground_truth, ground_truths, hypotheses)
                cnt += 1
    
    #error = wer(ground_truths, hypotheses, truth_transform=transformation, hypothesis_transform=transformation)
    error = wer(ground_truths, hypotheses)
    print('Processed %d files. WER is %f' % (cnt, error))

if __name__ == '__main__':
    # LJ
    synthesis_dir = 'exp/tts_sls_train_transformer-guided_attn_v1_raw_phn_none-w2vU2.0_v1/decode_transformer_valid.loss.ave_parallel_wavegan.v3/alex_test_nopunc/wav/' # WER is 0.216686
    synthesis_dir = 'exp/tts_sls_train_transformer-guided_attn_v1_raw_phn_none-w2vU2.0_v1/decode_transformer_valid.loss.ave_parallel_wavegan.v3/w2vU2.0_v1_test/wav/' # WER is 0.219812
    synthesis_dir = 'exp/tts_sls_train_transformer-guided_attn_v1_raw_phn_none-nopunc/decode_transformer_valid.loss.ave_parallel_wavegan.v3/alex_test_nopunc/wav/' # WER is 0.191676
    synthesis_dir = '/data/sls/temp/clai24/data/LJSpeech-1.1/wavs' # WER is 0.180344


    # LibriTTS
    synthesis_dir = 'downloads/LibriTTS/test-clean/' # WER is 0.233956
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-nopunc-spkfinetune-4535/decode_transformer_valid.loss.ave_hifigan/alex_test-clean_nopunc-4535/wav/' # WER is 0.263571
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1-spkfinetune-4535/decode_transformer_valid.loss.ave_hifigan/alex_test-clean_nopunc-4535/wav/' # WER is 0.311837
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-nopunc-spkfinetune-8699/decode_transformer_valid.loss.ave_hifigan/alex_test-clean_nopunc-8699/wav/' # WER is 0.213165
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1-spkfinetune-8699/decode_transformer_valid.loss.ave_hifigan/alex_test-clean_nopunc-8699/wav/' # WER is 0.254646
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-nopunc-spkfinetune-3922/decode_transformer_valid.loss.ave_hifigan/alex_test-clean_nopunc-3922/wav/' # WER is 0.291236
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1-spkfinetune-3922/decode_transformer_valid.loss.ave_hifigan/alex_test-clean_nopunc-3922/wav/' # WER is 0.332851
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-nopunc-spkfinetune-6209/decode_transformer_valid.loss.ave_hifigan/alex_test-clean_nopunc-6209/wav/' # WER is 0.302979
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1-spkfinetune-6209/decode_transformer_valid.loss.ave_hifigan/alex_test-clean_nopunc-6209/wav/' # WER is 0.362555
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-nopunc-spkfinetune-3638/decode_transformer_valid.loss.ave_hifigan/alex_test-clean_nopunc-3638/wav/' # WER is 0.223315
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1-spkfinetune-3638/decode_transformer_valid.loss.ave_hifigan/alex_test-clean_nopunc-3638/wav/' # WER is 0.286634
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-nopunc-spkfinetune-6701/decode_transformer_valid.loss.ave_hifigan/alex_test-clean_nopunc-6701/wav/' # WER is 0.226457
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1-spkfinetune-6701/decode_transformer_valid.loss.ave_hifigan/alex_test-clean_nopunc-6701/wav/' # WER is 0.268139

    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1/decode_speaker6209_transformer_valid.loss.best_hifigan/alex_test-clean_nopunc-6209/wav/' # WER is 0.468413
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1/decode_speaker3638_transformer_valid.loss.best_hifigan/alex_test-clean_nopunc-3638/wav/' # WER is 0.349931
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1/decode_speaker6701_transformer_valid.loss.best_hifigan/alex_test-clean_nopunc-6701/wav/' # WER is 0.323224
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1/decode_speaker3922_transformer_valid.loss.best_hifigan/alex_test-clean_nopunc-3922/wav/' # WER is 0.388738
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1/decode_speaker4535_transformer_valid.loss.best_hifigan/alex_test-clean_nopunc-4535/wav/' # WER is 0.325709
    synthesis_dir = 'exp/tts_sls_train_gst+xvector_transformer_raw_phn_none-w2vU2.0_v1/decode_speaker8699_transformer_valid.loss.best_hifigan/alex_test-clean_nopunc-8699/wav/' # WER is 0.326600

    synthesis_dir = 'downloads/LibriTTS/train-clean-360/8699' # WER is 0.248058
    synthesis_dir = 'downloads/LibriTTS/train-clean-360/4535' # WER is 0.273979
    synthesis_dir = 'downloads/LibriTTS/train-clean-360/6701' # WER is 0.248402
    synthesis_dir = 'downloads/LibriTTS/train-clean-360/3638' # WER is 0.217765
    synthesis_dir = 'downloads/LibriTTS/train-clean-360/3922' # WER is 0.263584
    synthesis_dir = 'downloads/LibriTTS/train-clean-100/6209' # WER is 0.252818
    run(synthesis_dir)
    #run('/data/sls/temp/clai24/data/LJSpeech-1.1/wavs')
    #run(sys.argv[1], sys.argv[2])


