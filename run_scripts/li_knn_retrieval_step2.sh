split=valid # 指定计算valid或test集特征
DATAPATH=/mnt/petrelfs/lijiaxing1/lijiaxing/Chinese-CLIP/data
dataset_name=Multimodal_Retrieval

python -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/old_MR_${split}_queries.jsonl.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/old_MR_${split}_queries.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/old_${split}_predictions.jsonl"