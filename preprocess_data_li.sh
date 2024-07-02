#!/usr/bin/env
set -x
DATAPATH=/mnt/petrelfs/lijiaxing1/lijiaxing/Chinese-CLIP/data
dataset_name=Multimodal_Retrieval


python cn_clip/preprocess/build_lmdb_dataset.py \
    --data_dir ${DATAPATH}/datasets/${dataset_name} \
    --splits train