#!/usr/bin/env bash

for LANG in en es ar ko; do
    
    TRAIN1=data/${LANG}100k_an_train.csv
    TRAIN2=data/${LANG}_ud_train.csv

    DEV1=data/${LANG}_ud.csv
    DEV2=data/${LANG}100k_an_dev.csv
    
    VOCAB=data/${LANG}_adj10k.txt
    VECTORS=~/data/wordvecs/wikivecs_aligned/wiki.${LANG}short.align.vec
    
    SEED=0

    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV1 --vocab $VOCAB > logs/${LANG}_bilinear_encoder_b64_wikitrain_uddev.csv
    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV2 --vocab $VOCAB > logs/${LANG}_bilinear_encoder_b64_wikitrain_uddev.csv

    python bilinear.py $VECTORS $TRAIN1 --batch_size 512 --seed $SEED --dev $DEV1 --vocab $VOCAB > logs/${LANG}_bilinear_encoder_b512_wikitrain_uddev.csv
    python bilinear.py $VECTORS $TRAIN1 --batch_size 512 --seed $SEED --dev $DEV2 --vocab $VOCAB > logs/${LANG}_bilinear_encoder_b512_wikitrain_uddev.csv

    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV1 --vocab $VOCAB --no_encoders > logs/${LANG}_bilinear_noencoder_b64_wikitrain_uddev.csv
    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV2 --vocab $VOCAB --no_encoders > logs/${LANG}_bilinear_noencoder_b64_wikitrain_uddev.csv

    python bilinear.py $VECTORS $TRAIN1 --batch_size 512 --seed $SEED --dev $DEV1 --vocab $VOCAB --no_encoders > logs/${LANG}_bilinear_noencoder_b512_wikitrain_uddev.csv
    python bilinear.py $VECTORS $TRAIN1 --batch_size 512 --seed $SEED --dev $DEV2 --vocab $VOCAB --no_encoders > logs/${LANG}_bilinear_noencoder_b512_wikitrain_uddev.csv

    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV1 --vocab $VOCAB --softmax > logs/${LANG}_softmax_encoder_b64_wikitrain_uddev.csv
    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV2 --vocab $VOCAB --softmax > logs/${LANG}_softmax_encoder_b64_wikitrain_uddev.csv

    python bilinear.py $VECTORS $TRAIN1 --batch_size 512 --seed $SEED --dev $DEV1 --vocab $VOCAB --softmax > logs/${LANG}_softmax_encoder_b512_wikitrain_uddev.csv
    python bilinear.py $VECTORS $TRAIN1 --batch_size 512 --seed $SEED --dev $DEV2 --vocab $VOCAB --softmax > logs/${LANG}_softmax_encoder_b512_wikitrain_uddev.csv

    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV1 --vocab $VOCAB --softmax --no_encoders > logs/${LANG}_softmax_noencoder_b64_wikitrain_uddev.csv
    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV2 --vocab $VOCAB --softmax --no_encoders > logs/${LANG}_softmax_noencoder_b64_wikitrain_uddev.csv

    python bilinear.py $VECTORS $TRAIN1 --batch_size 512 --seed $SEED --dev $DEV1 --vocab $VOCAB --softmax --no_encoders > logs/${LANG}_softmax_noencoder_b512_wikitrain_uddev.csv
    python bilinear.py $VECTORS $TRAIN1 --batch_size 512 --seed $SEED --dev $DEV2 --vocab $VOCAB --softmax --no_encoders > logs/${LANG}_softmax_noencoder_b512_wikitrain_uddev.csv

    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV1 --vocab $VOCAB --tie_params > logs/${LANG}_bilinear_encoder_b64_wikitrain_uddev_tied.csv
    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV2 --vocab $VOCAB --tie_params > logs/${LANG}_bilinear_encoder_b64_wikitrain_uddev_tied.csv

    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV1 > logs/${LANG}_bilinear_encoder_b64_wikitrain_uddev_vw.csv
    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV2 > logs/${LANG}_bilinear_encoder_b64_wikitrain_uddev_vw.csv
    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV1 --batch_size 512 > logs/${LANG}_bilinear_encoder_b512_wikitrain_uddev_vw.csv
    python bilinear.py $VECTORS $TRAIN1 --seed $SEED --dev $DEV2 --batch_size 512 > logs/${LANG}_bilinear_encoder_b512_wikitrain_uddev_vw.csv

done
