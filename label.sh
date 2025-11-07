cd src &&

TIMESTAMP=2025-11-06_17-11-12

echo ../logs/$TIMESTAMP/train_config.yaml
python hlabel.py \
    --label_config=../logs/$TIMESTAMP/train-config.yaml \
    --disable_segment
