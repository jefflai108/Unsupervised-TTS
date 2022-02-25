import os
import argparse

WAVS_DIR = 'downloads/LJSpeech-1.1/wavs/'

def create_data_dir(phn_seq_file, 
                   output_wav_scp_file, 
                   output_utt2spk_file):
    with open(phn_seq_file, 'r') as f: 
        content = f.readlines()
    content = [x.strip('\n') for x in content]
    id_seq  = [x.split()[0] for x in content]
    with open(output_wav_scp_file, 'w') as f: 
        for id in id_seq: 
            wav_path = os.path.join(WAVS_DIR, id + '.wav')
            f.write('%s %s\n' % (id, wav_path))

    with open(output_utt2spk_file, 'w') as f: 
        for id in id_seq: 
            f.write('%s %s\n' % (id, 'LJ'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_data_dir', type=str)
    parser.add_argument('--alex_data_dir', type=str)
    parser.add_argument('--data_split', type=str)
    args = parser.parse_args()

    create_data_dir(os.path.join(args.alex_data_dir, args.data_split + '.phn'), 
                    os.path.join(args.output_data_dir, 'wav.scp'),
                    os.path.join(args.output_data_dir, 'utt2spk'))
