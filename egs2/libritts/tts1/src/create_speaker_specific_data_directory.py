import numpy as np 
import os.path as osp
import argparse

TRAIN_SPK2UTT = 'data/alex_train-clean-460_w2vU2.0_v1/spk2utt'
DATA_FILES = ['wav.scp', 
              'text', 
              'spk2utt', 
              'spk2gender']

def random_select_spek(num_of_selected_speakers=6):
    content = read_file(TRAIN_SPK2UTT)
    with open(TRAIN_SPK2UTT, 'r') as f: 
        content = f.readlines()
    content = [x.strip('\n') for x in content]
    content = {x.split()[0]:len(x.split()[1:]) for x in content}
    from collections import defaultdict
    spk_content = defaultdict(int)
    for k,v in content.items(): 
        spk_id = k.split('_')[0]
        spk_content[spk_id] += v
    spk_content = [(k,v) for k,v in spk_content.items()]
    spk_content = sorted(spk_content, key=lambda x:x[1], reverse=True)[:num_of_selected_speakers]
    print(spk_content)

def read_file(fname): 
    with open(fname, 'r') as f: 
        content = f.readlines()
    content = [x.strip('\n') for x in content]

    return content 

def create_new_data_files(target_speaker, orig_train_data_dir, target_train_data_dir, target_valid_data_dir):
    orig_train_wav_scps = read_file(osp.join(orig_train_data_dir, 'wav.scp'))
    orig_train_texts = read_file(osp.join(orig_train_data_dir, 'text'))
    orig_train_spk2utts = read_file(osp.join(orig_train_data_dir, 'spk2utt'))
    orig_train_spk2genders = read_file(osp.join(orig_train_data_dir, 'spk2gender'))
   
    total_wav_scp = []
    for orig_train_wav_scp in orig_train_wav_scps: 
        if target_speaker == orig_train_wav_scp.split()[0].split('_')[0]: 
            total_wav_scp.append(orig_train_wav_scp)
    print('total training utterances for speaker %s is %d' % (target_speaker, len(total_wav_scp)))
    train_wav_scps = total_wav_scp[:round(len(total_wav_scp) * 0.8)]
    valid_wav_scps = total_wav_scp[round(len(total_wav_scp) * 0.8):]

    with open(osp.join(target_train_data_dir, 'wav.scp'), 'w') as f: 
        for train_wav_scp in train_wav_scps:
            f.write('%s\n' % train_wav_scp)

    with open(osp.join(target_valid_data_dir, 'wav.scp'), 'w') as f: 
        for valid_wav_scp in valid_wav_scps:
            f.write('%s\n' % valid_wav_scp)
   
    with open(osp.join(target_train_data_dir, 'spk2gender'), 'w') as f:
        for orig_train_spk2gender in orig_train_spk2genders:
            if orig_train_spk2gender.split()[0].split('_')[0] == target_speaker: 
                f.write('%s %s\n' % (target_speaker, orig_train_spk2gender.split()[1]))
                break

    with open(osp.join(target_valid_data_dir, 'spk2gender'), 'w') as f:
        for orig_train_spk2gender in orig_train_spk2genders:
            if orig_train_spk2gender.split()[0].split('_')[0] == target_speaker: 
                f.write('%s %s\n' % (target_speaker, orig_train_spk2gender.split()[1]))
                break

    with open(osp.join(target_train_data_dir, 'spk2utt'), 'w') as f:
        train_uttids = []
        for train_wav_scp in train_wav_scps:
            train_uttid = train_wav_scp.split()[0]
            assert target_speaker in train_uttid
            train_uttids.append(train_uttid)
        f.write('%s %s\n' % (target_speaker, ' '.join(train_uttids)))

    with open(osp.join(target_valid_data_dir, 'spk2utt'), 'w') as f:
        valid_uttids = []
        for valid_wav_scp in valid_wav_scps:
            valid_uttid = valid_wav_scp.split()[0]
            assert target_speaker in valid_uttid
            valid_uttids.append(valid_uttid)
        f.write('%s %s\n' % (target_speaker, ' '.join(valid_uttids)))
    
    with open(osp.join(target_train_data_dir, 'text'), 'w') as f: 
        for orig_train_text in orig_train_texts: 
            if orig_train_text.split()[0] in train_uttids: 
                f.write('%s\n' % orig_train_text)

    with open(osp.join(target_valid_data_dir, 'text'), 'w') as f: 
        for orig_train_text in orig_train_texts: 
            if orig_train_text.split()[0] in valid_uttids: 
                f.write('%s\n' % orig_train_text)

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_speaker',type=str)
    parser.add_argument('--orig_train_data_dir', type=str)
    parser.add_argument('--target_train_data_dir', type=str)
    parser.add_argument('--target_valid_data_dir', type=str)
    args = parser.parse_args()
    #random_select_spek()
    create_new_data_files(args.target_speaker, args.orig_train_data_dir, args.target_train_data_dir, args.target_valid_data_dir)

