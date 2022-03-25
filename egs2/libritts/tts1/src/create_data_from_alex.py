import os
import argparse

def create_data_dir(phn_seq_file, text_file):
    with open(phn_seq_file, 'r') as f: 
        content = f.readlines()
    content = [x.strip('\n') for x in content]
    id2phn_seq = {x.split()[0]:' '.join(x.split()[1:]) for x in content}

    with open(text_file, 'r') as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content]
    id_seq  = [x.split()[0] for x in content]

    with open(text_file, 'w') as f: 
        for id in id_seq: 
            f.write('%s %s\n' % (id, id2phn_seq[id]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phn_file', type=str)
    parser.add_argument('--text_file', type=str)
    args = parser.parse_args()

    create_data_dir(args.phn_file, args.text_file)

