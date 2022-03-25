import os.path as osp
import numpy as np 
import argparse

def fix_dump(orig_xvector_scp, target_xvector_scp, target_speaker_train_xvector_scp):
    with open(target_speaker_train_xvector_scp, 'r') as f: 
        spk_wise_xvectors = f.readlines()
    spk_wise_xvectors = [x.strip('\n') for x in spk_wise_xvectors]
    spk_wise_xvectors = {x.split()[0]: x.split()[1] for x in spk_wise_xvectors}

    with open(orig_xvector_scp, 'r') as f:
        utt_wise_xvectors = f.readlines()
    utts = [x.strip('\n').split()[0] for x in utt_wise_xvectors]

    sampled_keys = np.random.choice(list(spk_wise_xvectors.keys()), size=len(utt_wise_xvectors), replace=True,)

    with open(target_xvector_scp, 'w') as f:
        for i, utt_id in enumerate(utts): 
            f.write('%s %s\n' %(utt_id, spk_wise_xvectors[sampled_keys[i]]))

def fix_data_dump(orig_data_dump_dir, target_train_data_dump_dir): 
    #with open(osp.join(orig_data_dump_dir, 'spk2gender'), 'r') as f: 


    with open(osp.join(target_train_data_dump_dir, 'spk2gender'), 'r') as f: 
        gender = f.readline()
    

def fix_utt2sid():  
    with open('dump/raw/alex_test-clean_nopunc/utt2sid', 'r') as f: 
        uttid = f.readlines()
    uttids = [x.strip('\n').split()[0] for x in uttid]
    with open('dump/raw/alex_test-clean_nopunc/utt2sid', 'w') as f: 
        for uttid in uttids:
            f.write('%s %d\n' % (uttid, 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_xvector_scp', type=str)
    parser.add_argument('--target_xvector_scp', type=str)
    parser.add_argument('--target_speaker_train_xvector_scp', type=str)
    parser.add_argument('--orig_data_dump_dir', type=str)
    parser.add_argument('--target_train_data_dump_dir', type=str)
    args = parser.parse_args()

    fix_dump(args.orig_xvector_scp, args.target_xvector_scp, args.target_speaker_train_xvector_scp)
    #fix_data_dump(args.orig_data_dump_dir, args.target_train_data_dump_dir)
