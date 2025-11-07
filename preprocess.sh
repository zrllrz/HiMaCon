cd preprocess &&

export HF_HOME=../huggingface

# Set the parent path where LIBERO is located
YOUR_LIBERO_PARENT_PATH=

python libero.py \
    --libero_org_path $YOUR_LIBERO_PARENT_PATH/LIBERO/libero/datasets/libero/libero_10 \
    --libero_tar_path ../libero_dataset \
    --chunk_size 64

