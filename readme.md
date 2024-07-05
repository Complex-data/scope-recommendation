# SFT stage
### SFT Train
```shell
python train.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_title64_t_0_q_llama7b_CR/ 
--backbone meta-llama/Llama-2-7b-hf 
--item_index title64_t 
--batch_size 1 
--topk 10 
--gpu cuda:0 
--clip_grad_norm 1.0 
--epoch 10 
--gen_max_length 512 
--quantization 
--lr 0.0005 
--gradient_accumulation_steps 16 
--train_stage SFT 
--SFT_actor_lora_r 16 
--log_to_file
--SFT_train_tasks SFTSeqRec,SFTControlRec,SFTPersonalControlRec,SFTCategoryRate
--SFT_val_tasks SFTValSeqRec,SFTValControlRec,SFTValPersonalControlRec
--FA2
```

### SFT Train Continue
```shell
python train.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_title64_t_0_q_llama7b_CR/ 
--backbone meta-llama/Llama-2-7b-hf 
--item_index title64_t 
--batch_size 1 
--topk 10 
--gpu cuda:0 
--clip_grad_norm 1.0 
--epoch 10 
--gen_max_length 512 
--quantization 
--lr 0.0005 
--warmup_ratio 0.025
--gradient_accumulation_steps 16 
--train_stage SFT 
--SFT_actor_lora_r 16 
--log_to_file
--SFT_load {xxx}
--SFT_train_tasks SFTSeqRec,SFTControlRec,SFTPersonalControlRec,SFTCategoryRate
--SFT_val_tasks SFTValSeqRec,SFTValControlRec,SFTValPersonalControlRec
--FA2
```

### SFT Test after Merge
```shell
python train.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_Title64T_0_Q_Llama7b_SC/ 
--backbone {xxx}
--item_index title64_t 
--test_batch_size 16 
--topk 10 
--gpu cuda:0 
--gen_max_length 512 
--quantization 
--train_stage SFT_Test
--SFT_actor_lora_r 0 
--SFT_test_task SFTTestSeqRec 
--backup_ip 0.0.0.0 
--FA2
```

### SFT Test before Merge
```shell
python train.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_Title64T_0_Q_Llama7b_SC/ 
--backbone meta-llama/Llama-2-7b-hf 
--item_index title64_t 
--test_batch_size 16 
--topk 10 
--gpu cuda:0 
--gen_max_length 512 
--quantization 
--train_stage SFT_Test 
--SFT_actor_lora_r 16 
--SFT_test_task SFTTestSeqRec 
--backup_ip 0.0.0.0
--SFT_load {xxx} 
--FA2
```


# SFT model Merge
```shell
python train.py 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_Title64T_0_Q_Llama7b/ 
--backbone meta-llama/Llama-2-7b-hf 
--item_index title64_t 
--gpu cuda:0 
--train_stage SFT_Merge 
--SFT_actor_lora_r 16 
--SFT_load snap/ICR_SubMovie_Title64T_0_Q_Llama7b/Epoch20_SFT
```


# New


## SASRec Server start
```shell
cd SASRec/
python cli.py --dataset sub_movie --port 12621
```

## SFT stage

### SFT train
```shell
nohup accelerate launch --num_processes 4 --gpu_ids 0,1,2,3 --main_process_port 29502 train.py --seed 0 --data_path data/dataset/sub_movie/ --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ --backbone snap/Llama-2-7b-hf-chat/ --item_index title64_t --batch_size 1 --topk 10 --clip_grad_norm 1.0 --epoch 40 --gen_max_length 512 --lr 0.0006 --gradient_accumulation_steps 12 --train_stage SFT --SFT_actor_lora_r 16 --warmup_ratio 0.0125 --val_batch_size 12 --SFT_train_tasks SFTSeqRec,SFTControlRec,SFTPersonalControlRec,SFTPersonalCategoryRate,SFTCategoryRate --SFT_val_tasks SFTTestSeqRec,SFTTestSeqRanking,SFT+TestPersonalControlRec,SFT-TestPersonalControlRec,SFTTestPersonalCategoryRateLP_50,SFTTestItemCount --backup_ip 0.0.0.0 --val_epoch 5 --share_chat_gpt_ratio 0.5 --FA2 --llama2_chat_template --idx --SFT_load snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/Epoch07_SFT > snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/output8-40.log &
```

### SFT merge
```shell
python train.py --backbone snap/Llama-2-7b-hf-chat/ --gpu cuda:0 --train_stage SFT_Merge --SFT_actor_lora_r 16 --output snap/ICR_SubMovie_Title64T_0_Q_Llama7b/ --SFT_load snap/ICR_SubMovie_Title64T_0_Q_Llama7b/Epoch20_SFT
```


### VLLM deploy
```shell
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --port 13579 --model snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/
```

### VLLM test
```shell
python task_test.py --SFT_test_task SFTTestSeqRec --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
python task_test.py --SFT_test_task SFTTestSeqRec --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 5
python task_test.py --SFT_test_task SFTTestSeqRanking --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 5
python task_test.py --SFT_test_task SFTTestSeqRanking --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 3
python task_test.py --SFT_test_task SFT+TestPersonalControlRec --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
python task_test.py --SFT_test_task SFT-TestPersonalControlRec --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
python task_test.py --SFT_test_task SFTTestPersonalCategoryRate_30% --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
python task_test.py --SFT_test_task SFTTestPersonalCategoryRate_50% --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
python task_test.py --SFT_test_task SFTTestPersonalCategoryRate_70% --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
```