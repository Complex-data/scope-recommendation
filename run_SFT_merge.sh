python train.py \
--backbone path/to/your/backbone_weights_huggingface_format \
--gpu cuda:7 \
--train_stage SFT_Merge \
--SFT_actor_lora_r 16 \
--user_control_symbol \
--output path/to/snap/test \
--SFT_load path/to/Epoch40_SFT

# --user_control_symbol \