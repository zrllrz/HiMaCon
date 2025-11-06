
# export HF_HOME=/path/to/your/custom/cache

cd preprocess &&

export HF_HOME=../huggingface
export CUDA_VISIBLE_DEVICES="6"

python libero.py \
    --libero_org_path /group/ycyang/rzliu/LIBERO/libero/datasets/libero/libero_10 \
    --libero_tar_path ../libero_dataset \
    --chunk_size 64

