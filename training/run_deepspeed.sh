accelerate launch --config_file "training/configs/deepspeed_config.yaml"  train.py \
--config "configs/gpt2_train.yaml" \
--dataset_path "../outputs"