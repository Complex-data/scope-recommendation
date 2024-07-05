CUDA_VISIBLE_DEVICES=6 nohup python train.py \
--seed 0 \
--data_path path/to/dataset/movies/ \
--backbone path/to/your/backbone_weights_huggingface_format \
--item_index title \
--test_batch_size 16 \
--topk 10 \
--gpu cuda:0 \
--gen_max_length 512 \
--train_stage SFT_Test \
--SFT_actor_lora_r 16 \
--SFT_test_task SFTTestSeqRec \
--backup_ip 0.0.0.0 \
--idx \
--user_control_symbol \
--use_CBS \
--SFT_load path/to/Epoch40_SFT> path/to/Epoch40_SFT_test_output.log 2>&1 &
# --FA2
# --use_CBS \
#--user_control_symbol \
# todo : 时间花费： cbs 前缀树 检索花费很多时间？ 返回candidate list 模型花费了很多时间？