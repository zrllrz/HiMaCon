cd src &&

export CUDA_VISIBLE_DEVICES="6"
python htrain.py --train_config=../train-configs/libero-90.yaml
