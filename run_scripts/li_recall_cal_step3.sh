
DATAPATH=/mnt/petrelfs/lijiaxing1/lijiaxing/Chinese-CLIP/data
dataset_name=Multimodal_Retrieval

split=valid # 指定计算valid或test集特征
python cn_clip/eval/evaluation.py \
    ${DATAPATH}/datasets/${dataset_name}/MR_${split}_queries.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl \
    old-output.json
cat old-output.json