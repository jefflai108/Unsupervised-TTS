#!/bin/bash 

for spk in 6209 4535 6701 3638 3922 8699 ; do 
    orig_dump_dir=dump/xvector/alex_test-clean_nopunc-${spk}
    cp ${orig_dump_dir}/xvector.scp ${orig_dump_dir}/orig_xvector.scp   
    cat dump/xvector/alex_train-clean-460_w2vU2.0_v1-${spk}-train/xvector.scp > ${orig_dump_dir}/target_speaker_train_xvector.scp 
    cat dump/xvector/alex_train-clean-460_w2vU2.0_v1-${spk}-valid/xvector.scp >> ${orig_dump_dir}/target_speaker_train_xvector.scp 
    
    cp dump/raw/alex_train-clean-460_w2vU2.0_v1-${spk}-train/spk2gender dump/raw/alex_test-clean_nopunc-${spk}/spk2gender

    python src/fix_test_set_xvector.py \
        --orig_xvector_scp ${orig_dump_dir}/orig_xvector.scp \
        --target_xvector_scp ${orig_dump_dir}/xvector.scp \
        --target_speaker_train_xvector_scp ${orig_dump_dir}/target_speaker_train_xvector.scp \
        --orig_data_dump_dir dump/raw/alex_test-clean_nopunc-${spk} \
        --target_train_data_dump_dir dump/raw/alex_train-clean-460_w2vU2.0_v1-${spk}-train
done 
