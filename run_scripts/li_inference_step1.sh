export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip


DATAPATH=/mnt/petrelfs/lijiaxing1/lijiaxing/Chinese-CLIP/data
dataset_name=Multimodal_Retrieval
split=test # 指定计算valid或test集特征
resume=/mnt/petrelfs/lijiaxing1/lijiaxing/Chinese-CLIP/data/experiments/li_finetune_6.1_2_vit-h-14_roberta-base_bs128_4gpu/checkpoints/best.pt



python -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/${dataset_name}/MR_${split}_queries.jsonl" \
    --img-batch-size=64 \
    --text-batch-size=64 \
    --context-length=52 \
    --resume=${resume} \
    --vision-model=ViT-H-14 \
    --text-model=RoBERTa-wwm-ext-large-chinese


python -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/MR_${split}_queries.jsonl.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/MR_${split}_queries.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_predictions_6.1.jsonl"

# split=valid # 指定计算valid或test集特征
# python cn_clip/eval/evaluation.py \
#     ${DATAPATH}/datasets/${dataset_name}/MR_${split}_queries.jsonl \
#     ${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl \
#     5.29_L_output.json
# cat 5.29_L_output.json

