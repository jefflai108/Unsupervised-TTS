#!/bin/bash 

# 100 (train) / 20 (valid)  of utt/speaker
#5393_19219
#6637_69607
#4137_11702
#4899_32637
#4297_13009
#1841_159771
#8194_89390
#7434_75939
#6371_63713
#7000_83708

# new ones (all males to balance the gender)
#5293_82020
#6701_71401
#7128_84121
#3185_163505
#1027_125147

# specified in the LibriTTS papers
# 19: 16 utt
# 103: 66 utt 
# 1841: 260 utt

# 204: 99 utt
# 1121: 122 utt
# 5717: 256 utt

# combine 6 speakers data dir into 1
if false; then 
target_speaker=6209_4535_6701_3638_3922_8699
for split in train valid; do
./utils/combine_data.sh data/alex_train-clean-460_w2vU2.0_v1-${target_speaker}-${split} \
                        data/alex_train-clean-460_w2vU2.0_v1-6209-${split} \
                        data/alex_train-clean-460_w2vU2.0_v1-4535-${split} \
                        data/alex_train-clean-460_w2vU2.0_v1-6701-${split} \
                        data/alex_train-clean-460_w2vU2.0_v1-3638-${split} \
                        data/alex_train-clean-460_w2vU2.0_v1-3922-${split} \
                        data/alex_train-clean-460_w2vU2.0_v1-8699-${split}
done
cp -r data/alex_test-clean_nopunc data/alex_test-clean_nopunc-${target_speaker} 
fi

# test
orig_test_dir=data/alex_test-clean_nopunc
for spk in 6209 4535 6701 3638 3922 8699 ; do 
    target_test_dir=data/alex_test-clean_nopunc-${spk}
    [ ! -d $target_test_dir ] && cp -r $orig_test_dir $target_test_dir 
    
done 
exit 0

# train/valid 
orig_data_dir=data/alex_train-clean-460_nopunc
for spk in 6209 4535 6701 3638 3922 8699 ; do 
    target_train_dir=data/alex_train-clean-460_nopunc-${spk}-train
    target_valid_dir=data/alex_train-clean-460_nopunc-${spk}-valid
    [ ! -d $target_train_dir ] && cp -r $orig_data_dir $target_train_dir 
    [ ! -d $target_valid_dir ] && cp -r $orig_data_dir $target_valid_dir
    python src/create_speaker_specific_data_directory.py \
        --target_speaker $spk \
        --orig_train_data_dir $orig_data_dir \
        --target_train_data_dir $target_train_dir \
        --target_valid_data_dir $target_valid_dir

    utt2spk=${target_train_dir}/utt2spk
    spk2utt=${target_train_dir}/spk2utt
    utils/spk2utt_to_utt2spk.pl ${spk2utt} > ${utt2spk}

    utt2spk=${target_valid_dir}/utt2spk
    spk2utt=${target_valid_dir}/spk2utt
    utils/spk2utt_to_utt2spk.pl ${spk2utt} > ${utt2spk}

    utils/fix_data_dir.sh $target_train_dir
    utils/fix_data_dir.sh $target_valid_dir
    utils/validate_data_dir.sh --no-feats $target_train_dir
    utils/validate_data_dir.sh --no-feats $target_valid_dir
done 
exit 0

# train/valid 
orig_data_dir=data/alex_train-clean-460_w2vU2.0_v1
for spk in 6209 4535 6701 3638 3922 8699 ; do 
    target_train_dir=data/alex_train-clean-460_w2vU2.0_v1-${spk}-train
    target_valid_dir=data/alex_train-clean-460_w2vU2.0_v1-${spk}-valid
    [ ! -d $target_train_dir ] && cp -r $orig_data_dir $target_train_dir 
    [ ! -d $target_valid_dir ] && cp -r $orig_data_dir $target_valid_dir
    python src/create_speaker_specific_data_directory.py \
        --target_speaker $spk \
        --orig_train_data_dir $orig_data_dir \
        --target_train_data_dir $target_train_dir \
        --target_valid_data_dir $target_valid_dir

    utt2spk=${target_train_dir}/utt2spk
    spk2utt=${target_train_dir}/spk2utt
    utils/spk2utt_to_utt2spk.pl ${spk2utt} > ${utt2spk}

    utt2spk=${target_valid_dir}/utt2spk
    spk2utt=${target_valid_dir}/spk2utt
    utils/spk2utt_to_utt2spk.pl ${spk2utt} > ${utt2spk}

    utils/fix_data_dir.sh $target_train_dir
    utils/fix_data_dir.sh $target_valid_dir
    utils/validate_data_dir.sh --no-feats $target_train_dir
    utils/validate_data_dir.sh --no-feats $target_valid_dir
done 


