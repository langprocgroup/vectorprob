#!/usr/bin/env bash

TRAIN=data/en_ud_train.csv
DEV1=data/en_ud_devtest.csv
DEV2=data/enwiki_dev.csv
VOCAB=data/en_adj10k.txt
SEED=0

python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --seed $SEED --dev $DEV1 --vocab $VOCAB > logs/en_bilinear_encoder_d0_udtrain_uddev.csv
python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --seed $SEED --dev $DEV2 --vocab $VOCAB > logs/en_bilinear_encoder_d0_udtrain_wikidev.csv

python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --dropout 0.1 --seed $SEED --dev $DEV1 --vocab $VOCAB > logs/en_bilinear_encoder_d01_udtrain_uddev.csv
python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --dropout 0.1 --seed $SEED --dev $DEV2 --vocab $VOCAB > logs/en_bilinear_encoder_d01_udtrain_wikidev.csv

python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --seed $SEED --dev $DEV1 --vocab $VOCAB --no_encoders > logs/en_bilinear_noencoder_d0_udtrain_uddev.csv
python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --seed $SEED --dev $DEV2 --vocab $VOCAB --no_encoders > logs/en_bilinear_noencoder_d0_udtrain_wikidev.csv

python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --dropout 0.1 --seed $SEED --dev $DEV1 --vocab $VOCAB --no_encoders > logs/en_bilinear_noencoder_d01_udtrain_uddev.csv
python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --dropout 0.1 --seed $SEED --dev $DEV2 --vocab $VOCAB --no_encoders > logs/en_bilinear_noencoder_d01_udtrain_wikidev.csv

python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --seed $SEED --dev $DEV1 --vocab $VOCAB --softmax > logs/en_softmax_encoder_d0_udtrain_uddev.csv
python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --seed $SEED --dev $DEV2 --vocab $VOCAB --softmax > logs/en_softmax_encoder_d0_udtrain_wikidev.csv

python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --dropout 0.1 --seed $SEED --dev $DEV1 --vocab $VOCAB --softmax > logs/en_softmax_encoder_d01_udtrain_uddev.csv
python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --dropout 0.1 --seed $SEED --dev $DEV2 --vocab $VOCAB --softmax > logs/en_softmax_encoder_d01_udtrain_wikidev.csv

python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --seed $SEED --dev $DEV1 --vocab $VOCAB --softmax --no_encoders > logs/en_softmax_noencoder_d0_udtrain_uddev.csv
python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --seed $SEED --dev $DEV2 --vocab $VOCAB --softmax --no_encoders > logs/en_softmax_noencoder_d0_udtrain_wikidev.csv

python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --dropout 0.1 --seed $SEED --dev $DEV1 --vocab $VOCAB --softmax --no_encoders > logs/en_softmax_noencoder_d01_udtrain_uddev.csv
python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --dropout 0.1 --seed $SEED --dev $DEV2 --vocab $VOCAB --softmax --no_encoders > logs/en_softmax_noencoder_d01_udtrain_wikidev.csv

python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --seed $SEED --dev $DEV1 --vocab $VOCAB --tie_params logs/en_bilinear_encoder_d0_udtrain_uddev_tied.csv
python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --seed $SEED --dev $DEV2 --vocab $VOCAB --tie_params logs/en_bilinear_encoder_d0_udtrain_wikidev_tied.csv

python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --seed $SEED --dev $DEV1 logs/en_bilinear_encoder_d0_udtrain_uddev_vw.csv
python bilinear.py /Users/canjo/data/wikivecs/wiki.enshort.align.vec $TRAIN --seed $SEED --dev $DEV2 logs/en_bilinear_encoder_d0_udtrain_wikidev_vw.csv
